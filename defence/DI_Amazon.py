#key functions are copied from https://github.com/cleverhans-lab/dataset-inference
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import mlconfig
from Transfer_Adv import Transfer_Untargeted, Transfer2
from utils import *
from models import *
from utils import test, train_sur
import matplotlib.pyplot as plt
# from train import train
 
def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def norms_linf_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]

def loss_mingd(preds, target):
    loss =  (preds.max(dim = 1)[0] - preds[torch.arange(preds.shape[0]),target]).mean()
    assert(loss >= 0)
    return loss

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, gap, k = 10) :
    #Check which all directions can still be increased such that
    #they haven't been clipped already and have scope of increasing
    # ipdb.set_trace()
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    # print (batch_size)
    neg1 = (grad < 0)*(X_curr <= gap)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)

def rand_steps(model, X, y, args, target = None):
    #optimized implementation to only query remaining points
    del target#The attack does not use the targets
    start = time.time()
    # is_training = model.training
    # model.eval()                    # Need to freeze the batch norm and dropouts
    
    #Define the Noise
    uni, std, scale = (0.01, 0.01, 0.02); steps = 10
    if args.dataset == "SVHN":
        uni, std, scale = 2*uni, 2*std, 2*scale; steps = 100
    noise_2 = lambda X: torch.normal(0, std, size=X.shape).cuda()
    noise_1 = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=X.shape)).float().to(X.device) 
    noise_inf = lambda X: torch.empty_like(X).uniform_(-uni,uni)

    noise_map = {"l1":noise_1, "l2":noise_2, "linf":noise_inf}
    mag = 1

    delta = noise_map[args.distance](X)
    delta_base = delta.clone()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)  
    loss = 0
    y=y.detach().cpu()
    with torch.no_grad():
        for t in range(steps):   
            if t>0: 
                preds = model(X_r+delta_r, y_r)
                new_remaining = (preds == y[remaining])
                remaining[remaining.clone()] = new_remaining
            else: 
                preds = model(X+delta, y)
                remaining = (preds == y)
                
            if remaining.sum() == 0: break

            X_r = X[remaining]; delta_r = delta[remaining]; y_r = y[remaining]
            # preds = model(X_r + delta_r, y_r)
            mag+=1; delta_r = delta_base[remaining]*mag
            # delta_r += noise_map[args.distance](delta_r)
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1-X_r) # clip X+delta_r[remaining] to [0,1]
            delta[remaining] = delta_r.detach()
            
        print(f"Number of steps = {t+1} | Failed to convert = {(model(X+delta, y)==y).sum().item()} | Time taken = {time.time() - start}")
    # if is_training:
    #     model.train()    
    return delta

def mingd(model, X, y, args, target):
    start = time.time()
    is_training = model.training
    model.eval()                    # Need to freeze the batch norm and dropouts
    criterion = loss_mingd
    norm_map = {"l1":norms_l1_squeezed, "l2":norms_l2_squeezed, "linf":norms_linf_squeezed}
    alpha_map = {"l1":args.alpha_l_1/args.k, "l2":args.alpha_l_2, "linf":args.alpha_l_inf}
    alpha = float(alpha_map[args.distance])

    delta = torch.zeros_like(X, requires_grad=False)    
    loss = 0
    for t in range(args.num_iter):        
        if t>0: 
            preds = model(X_r+delta_r)
            new_remaining = (preds.max(1)[1] != target[remaining])
            remaining_temp = remaining.clone()
            remaining[remaining.clone()] = new_remaining
        else: 
            preds = model(X+delta)
            remaining = (preds.max(1)[1] != target)
            
        if remaining.sum() == 0: break

        X_r = X[remaining]; delta_r = delta[remaining]
        delta_r.requires_grad = True
        preds = model(X_r + delta_r)
        loss = -1* loss_mingd(preds, target[remaining])
        # print(t, loss, remaining.sum().item())
        loss.backward()
        grads = delta_r.grad.detach()
        if args.distance == "linf":
            delta_r.data += alpha * grads.sign()
        elif args.distance == "l2":
            delta_r.data += alpha * (grads / norms_l2(grads + 1e-12))
        elif args.distance == "l1":
            delta_r.data += alpha * l1_dir_topk(grads, delta_r.data, X_r, args.gap, args.k)
        delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1-X_r) # clip X+delta_r[remaining] to [0,1]
        delta_r.grad.zero_()
        delta[remaining] = delta_r.detach()
        
    print(f"Number of steps = {t+1} | Failed to convert = {(model(X+delta).max(1)[1]!=target).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()    
    return delta

def get_random_label_only(args, loader, model, num_images = 500):
    print("Getting random attacks")
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(2): #5 random starts
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                # preds = model(X)
                targets = None
                delta = rand_steps(model, X, y, args, target = targets)
                # yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d

def get_mingd_vulnerability(args, loader, model, num_images = 500):
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    ex_skipped = 0
    for i,batch in enumerate(loader):
        if args.regressor_embed == 1: ##We need an extra set of `distinct images for training the confidence regressor
            if(ex_skipped < num_images):
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
        for j,distance in enumerate(["linf", "l2", "l1"]):
            temp_list = []
            for target_i in range(args.num_classes):
                X,y = batch[0].to(device), batch[1].to(device) 
                args.distance = distance
                # args.lamb = 0.0001
                delta = mingd(model, X, y, args, target = y*0 + target_i)
                yp = model(X+delta) 
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta)
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            # temp_dist = [batch_size, num_classes)]
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist) 
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]    
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)
        
    return full_d

def feature_extractor(args, config):
    train_loader, test_loader, watermarkloader = load_data(config, adv=True)
    if args.independent==True:
        # test_loader, train_loader, watermarkloader = load_data(config, adv=True)
        student = config.ind_model()
        train_sur(config,student,test_loader,50, f"defences/DI/models/{args.dataset}/model_ind/log.txt")
        test_acc = test(student, train_loader)
        with open(f"defences/DI/models/{args.dataset}/model_ind/log.txt","a+") as f:
            f.write("Test: {}\n".format(test_acc))
        torch.save(student.state_dict(), f"defences/DI/models/{args.dataset}/model_ind/final.pt")
    if args.Amazon == True:
        import boto3
        client=boto3.client('rekognition')
        def student(batch_x, target):
            label = torch.zeros(len(batch_x)).long()
            for idx,x in enumerate(batch_x):
                plt.imsave(args.model_dir+'/temp.jpg',((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0]))
                with open(args.model_dir+'/temp.jpg','rb') as img:
                    response = client.detect_faces(Image={'Bytes': img.read()},    Attributes=[
                    'ALL'
                    ])
                    if len(response['FaceDetails'])>0:
                        label[idx]= int(response['FaceDetails'][0]['Gender']['Value']=='Male')
                    else:
                        label[idx]= 1-target[idx]
            return label
    if args.extract:
        teacher = config.model()
        teacher.load_state_dict(torch.load(f"defences/DI/models/{args.dataset}/model_teacher/final.pt")) 
        extract_loader = extract_dataset(teacher, train_loader,100)
        ext_model = config.model()
        train_sur(config,ext_model,extract_loader,50, f"defences/DI/models/{args.dataset}/model_extract/log.txt")
        test_acc = test(ext_model, test_loader)
        with open(f"defences/DI/models/{args.dataset}/model_extract/log.txt","a+") as f:
            f.write("Test: {}\n".format(test_acc))
        torch.save(ext_model.state_dict(), f"defences/DI/models/{args.dataset}/model_extract/final.pt")
    if args.finetune:
        fint_model = config.model()
        fint_model.load_state_dict(torch.load(f"defences/DI/models/{args.dataset}/model_teacher/final.pt"))
        extract_loader = extract_dataset(fint_model, train_loader,100)
        train_sur(config,fint_model,extract_loader,5,f"defences/DI/models/{args.dataset}/model_fine-tune/log.txt" )
        test_acc = test(fint_model, test_loader)
        with open(f"defences/DI/models/{args.dataset}/model_fine-tune/log.txt","a+") as f:
            f.write("Test: {}\n".format(test_acc))
        torch.save(fint_model.state_dict(), f"defences/DI/models/{args.dataset}/model_fine-tune/final.pt")

    if args.model_id == 'suspect' or args.model_id == 'suspect_same_data' or args.model_id=='suspect2':
        student = config.ind_model() #teacher is not needed
    elif args.Amazon == False:
        student = config.model()
    location = f"{args.model_dir}/final.pt"
    if args.Amazon == False:
        try:
            student = student.to(args.device)
            student.load_state_dict(torch.load(location, map_location = args.device)) 
        except:
            student = student.to(args.device)
            student = nn.DataParallel(student).to(args.device)
            student.load_state_dict(torch.load(location, map_location = args.device))   
    
        student.eval()
        test_acc = test(student, test_loader)
        print(f'Model: {args.model_dir} | \t Test Acc: {test_acc:.3f}')   
    if args.adv_gen:
        victim_model = config.model()
        victim_model.load_state_dict(torch.load(config.modeldir+"victim/"+"{}{}_victim_model.pth".format('half', 0)))
        test(victim_model,test_loader)
        ref_models = load_model(config, 0)

        attack = Transfer2(victim_model, ref_models, config)
        water_adv = []
        y_adv = []
        acc = 0
        num = 0

        for batch_idx, (input, label) in enumerate(tqdm(watermarkloader, unit="images", desc="Training adv exp for (watermark)"), 0):
            trigger, target = attack(images=input,labels=label,given_labels=label)
            trigger, target = trigger.cuda(), target.cuda().long()
            pred =victim_model(trigger).argmax(axis=1)
            # print(target, pred)
            acc += sum(pred==target).item()
            num += target.size(0)
            water_adv.append(trigger.cpu().numpy())
            y_adv.append(target.cpu().numpy())

        acc = acc/num
        print("adv acc: ", acc)

        # print("Gen adv watermark time: ", time.time()- start)
        water_adv = np.concatenate(water_adv*10)
        y_adv = np.concatenate(y_adv*10)
        water_gen = SimpleDataset([(img,label) for img, label in zip(water_adv, y_adv)])
        watermarkloader_a = data.DataLoader(water_gen, batch_size=config.train.batch_size, shuffle=False)
        torch.save(watermarkloader_a, f"defences/DI/models/{args.dataset}/model_teacher/wm_loader.pt")
    # _, train_acc  = epoch_test(args, train_loader, student)
    else:
        watermarkloader_a = torch.load(f"defences/DI/models/{args.dataset}/model_teacher/wm_loader.pt")
 
    
    mapping = {'pgd':None, 'topgd':None, 'mingd':get_mingd_vulnerability, 'rand': get_random_label_only}

    func = mapping[args.feature_type]

    # test_d = func(args, test_loader, student)
    # print(test_d)
    # torch.save(test_d, f"{args.file_dir}/test_{args.feature_type}_vulnerability_2.pt")
    # torch.save(test_d, f"{args.file_dir_adv}/test_{args.feature_type}_vulnerability_2.pt")
    if args.adv == True:
        train_d_adv = func(args, watermarkloader_a, student)   
        print(train_d_adv)
        torch.save(train_d_adv, f"{args.file_dir_adv}/train_{args.feature_type}_vulnerability_2.pt")
    # train_d = func(args, train_loader, student)
    # print(train_d)
    # torch.save(train_d, f"{args.file_dir}/train_{args.feature_type}_vulnerability_2.pt")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/celebA/train/resnet_di.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--save_dir', type=str, default="defences/celeba/DI")   
    parser.add_argument('--model_id', type=str, default='Amazon')
    parser.add_argument('--feature_type', type=str, default='rand')
    parser.add_argument('--adv_gen', type=bool, default=False)
    parser.add_argument('--adv', type=bool, default=True)
    parser.add_argument('--independent', type=bool, default=False)
    parser.add_argument('--extract', type=bool, default=False)
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--Amazon', type=bool, default=True)
    parser.add_argument('--num_iter', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--distance", help="Type of Adversarial Perturbation", type=str)#, choices = ["linf", "l1", "l2", "vanilla"])
    parser.add_argument("--randomize", help = "For the individual attacks", type = int, default = 0, choices = [0,1,2])
    parser.add_argument("--alpha_l_1", help = "Step Size for L1 attacks", type = float, default = 1.0)
    parser.add_argument("--alpha_l_2", help = "Step Size for L2 attacks", type = float, default = 0.01)
    parser.add_argument("--alpha_l_inf", help = "Step Size for Linf attacks", type = float, default = 0.001)

    parser.add_argument("--epsilon_l_1", help = "Step Size for L1 attacks", type = float, default = 12)
    parser.add_argument("--epsilon_l_2", help = "Epsilon Radius for L2 attacks", type = float, default = 0.5)
    parser.add_argument("--epsilon_l_inf", help = "Epsilon Radius for Linf attacks", type = float, default = 8/255.)
    parser.add_argument("--restarts", help = "Random Restarts", type = int, default = 1)
    parser.add_argument("--smallest_adv", help = "Early Stop on finding adv", type = int, default = 1)
    parser.add_argument("--gap", help = "For L1 attack", type = float, default = 0.001)
    parser.add_argument("--k", help = "For L1 attack", type = int, default = 100)
    parser.add_argument("--regressor_embed", help = "Victim Embeddings for training regressor", type = int, default = 0, choices = [0,1])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = mlconfig.load(args.config)
    args.batch_size = config.train.batch_size
    args.dataset = config.dataset.name
    print(args, config)
    device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")
    root = f"{args.save_dir}/models/{args.dataset}"
    model_dir = f"{root}/model_{args.model_id}"; print("Model Directory:", model_dir); args.model_dir = model_dir
    root = f"{args.save_dir}/files/{args.dataset}"  
    file_dir = f"{root}/model_{args.model_id}" 
    print("File Directory:", file_dir) ; args.file_dir = file_dir ; args.file_dir_adv = file_dir+"_adv"
    if(not os.path.exists(file_dir)):
        os.makedirs(file_dir)
    if(not os.path.exists(args.file_dir_adv)):
        os.makedirs(args.file_dir_adv)
    if config.dataset.name=='CelebA':
        args.num_classes = config.model.num_attributes+1
    else:
        args.num_classes = config.model.num_classes
    feature_extractor(args, config)