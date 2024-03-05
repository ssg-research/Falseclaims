import argparse
import mlconfig
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# import os
from utils import *
from models import *
from Transfer_Adv import Transfer, Transfer2, Transfer_Untargeted
from train import train
from utils import test

import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/cifar10/train/resnet_lukas.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--save_dir', type=str, default='defences/cifar10/lukas')   
    parser.add_argument('--ind_resume', type=bool, default=True)
    parser.add_argument('--inds_resume', type=bool, default=True)
    parser.add_argument('--vic_resume', type=bool, default=False)
    parser.add_argument('--new_lukas', type=bool, default=False)
    parser.add_argument('--adv', default=True)
    parser.add_argument('--model_count', type=int, default=1)

    return parser.parse_args()



def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    print(args, config)
    if args.adv == True:
        adv = 'half'
    else:
        adv = ''

    print("Load Ref Models...")
    ref_models = load_model(config, args.model_count)
    print("Ref Models Loaded.")
    # torch.manual_seed(777)

    print("Load Surrgoate Models...")
    sur_models = load_sur_model(config, 0)
    print("Surrgoate Models Loaded.")
    
    
    print("Initial Victim/Ind Models...")
    victim_model = config.model()
    victim_model.load_state_dict(torch.load(config.modeldir+"victim/{}{}_victim_model.pth".format(adv,args.model_count)))
    print("Victim models loaded.")
    victim_model = victim_model.cuda()
    victim_model.eval()

    # if not args.inds_resume:
    #     print("Train Ind Models... ")
    #     ind_models = []
    #     ind_model = config.ind_model()
    #     for i in range(config.num_ind_model):
    #         ind_model_temp = config.ind_model()
    #         ind_models.append(ind_model_temp)
    #     for i in range(config.num_ind_model):
    #         ind_model_temp = config.ind_model()
    #         ind_models.append(ind_model_temp)
    #     if not os.path.isdir(config.logdir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name)):
    #         os.makedirs(config.logdir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name))
    #     if not os.path.isdir(config.modeldir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name)):
    #         os.makedirs(config.modeldir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name))
    #     for i in range(config.num_ind_model):
    #         if args.adv == True:
    #             trainloader, _, _ = load_data(config, adv=not args.adv)
    #         else:
    #             trainloader, _, _ = load_data(config, adv=args.adv)
    #         train(config, ind_models[i], trainloader, config.logdir+"ind/{}{}_{}/{}{}_{}_ind{}.log".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i))
    #         test(ind_models[i], testloader, config.logdir+"ind/{}{}_{}/{}{}_{}_ind{}.log".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i))
    #         torch.save(ind_models[i].state_dict(), config.modeldir+"ind/{}{}_{}/{}{}_{}_ind_model{}.pth".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i))
    #         ind_models[i].eval()

    # else:
    #     print("Load Ind Models... ")
    #     ind_models = []
    #     for i in range(config.num_ind_model):
    #         ind_model_temp = config.ind_model()
    #         ind_models.append(ind_model_temp)
    #     for i in range(config.num_ind_model):
    #         ind_models[i].load_state_dict(torch.load(config.modeldir+"ind/{}{}_{}/{}{}_{}_ind_model{}.pth".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i)))
    #         ind_models[i].eval()
    #         ind_models[i] = ind_models[i].cuda()
    #     print("Ind models loaded.")

    if not args.ind_resume:
        ind_model = config.ind_model()
        # torch.manual_seed(667)
        print("Load Ind Data...")        
        trainloader, _, _ = load_data(config, adv=not args.adv)
        print("Load Victim Data finished.")
        print("Train Ind Model... ")
        train(config, ind_model, trainloader, config.logdir+"ind/{}{}_{}_ind.log".format(adv, args.model_count,config.ind_model.name))
        test(ind_model, testloader, config.logdir+"ind/{}{}_{}_ind.log".format(adv,args.model_count,config.ind_model.name))
        torch.save(ind_model.state_dict(), config.modeldir+"ind/"+"{}{}_{}_ind_model.pth".format(adv,args.model_count,config.ind_model.name))
    else:
        ind_model = config.ind_model()
        print("Load Ind Models... ")
        ind_model.load_state_dict(torch.load(config.modeldir+"ind/"+"{}{}_{}_ind_model.pth".format(adv,args.model_count,config.ind_model.name)))
        print("Ind models loaded.")


    print("Load Victim Data...")
    trainloader, testloader, watermarkloader = load_data(config, adv=args.adv)
    print("Load Victim Data finished.")

    # test_watermark(victim_model, testloader)
    start = time.time()
    def lukas(images, labels, config, victim_model, ref_models, sur_models, max_iter=20):
        images = images.cuda()
        labels = labels.cuda()
        criterion = nn.CrossEntropyLoss()
        targets = torch.zeros_like(labels)
        used_img_id = 0
        used_imgs = torch.zeros([config.used_wm_len,images.shape[1],images.shape[2],images.shape[3]])
        used_targets = torch.zeros([config.used_wm_len])

        victim_model.eval()

        img_loop = tqdm(range(images.shape[0]),desc="Transfer Generating...")

        for b in img_loop:
            image = images[b:b+1,:,:,:].detach().clone()
            # image = arctanh(image)
            # image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
            target = labels[b] ##改成非target label

            while target == labels[b].cpu():
                if config.dataset.name == 'CelebA':
                    target = torch.randint(config.model.num_attributes+1, (1,))
                else:
                    target = torch.randint(config.model.num_classes, (1,))
                # target = torch.argsort(pred.cpu().detach())[-2].unsqueeze(0)
            targets[b] = target.detach()
            noise = image.detach().clone()
            noise.requires_grad_()
            optimizer = torch.optim.SGD([noise,], lr=0.2)
            
            loop = tqdm(range(max_iter),desc="Adv generating.", ncols=150)
            max_noise = torch.ones_like(image) * (config.eps/255)*(torch.max(image)-torch.min(image))
            
            for i in loop:
                x = torch.clamp(noise, image-max_noise, image+max_noise)
                output =victim_model(x)[0]
                def closure():
                    optimizer.zero_grad()
                    x = torch.clamp(noise, image-max_noise, image+max_noise)
                    output = victim_model(x)[0]
                    loss = 0
                    # loss_0 = criterion(output.unsqueeze(0), target.cuda())
                    # loss_ref = torch.zeros_like(loss_0)
                    ref = torch.zeros_like(output)
                    for ref_model in ref_models:
                        ref_model.eval()
                        ref_out = ref_model(x)[0]
                        ref += ref_out/len(ref_models)
                        # loss_ref = loss_ref + criterion(ref_out.unsqueeze(0), target.cuda())/len(ref_models)
                    
                    # loss_sur = torch.zeros_like(loss_0)
                    sur = torch.zeros_like(output)                
                    for sur_model in sur_models:
                        sur_model.eval()
                        sur_out = sur_model(x)[0]
                        sur += sur_out/len(sur_models)
                        # loss_sur = loss_sur + criterion(sur_out.unsqueeze(0), target.cuda())/len(ref_models)
                    
                    ensemble_out = sur * (torch.ones_like(ref)- (ref-ref.min())/(ref.max()-ref.min()))
                    
                    first_loss = criterion(ensemble_out.unsqueeze(0), target.cuda())
                    # second_loss = criterion(output.unsqueeze(0), victim_model(image).argmax(1))
                    # score_out = F.log_softmax(output.unsqueeze(0),dim=1)
                    # second_loss = - torch.sum(score_out*torch.softmax(victim_model(image)[0],0))/1
                    second_loss = criterion(output.unsqueeze(0), target.cuda())
                    # thrid_loss = -torch.sum(score_out*torch.softmax(sur,0))/1
                    thrid_loss = criterion(sur.unsqueeze(0), target.cuda())
                    
                    loss = first_loss+second_loss+thrid_loss
                    loop.set_postfix(loss=loss.item(),label=labels[b].item(), target=target.item(),pred=output.argmax().item())
                    loop.set_description("Idx:{}, Iter:{}".format(b, i))
                    loss.backward()
                    return torch.softmax(ensemble_out,0)
                ensemble_out = optimizer.step(closure)

            x = torch.clamp(noise, image-max_noise, image+max_noise)
            output =torch.softmax(victim_model(x)[0],0)   
            if ensemble_out[target] < 0.95 and output.argmax().detach().cpu().item()!=target.item():
                continue
         
            used_imgs[used_img_id:used_img_id+1,:,:,:] = x.detach().clone()
            used_targets[used_img_id]=torch.LongTensor(target)
            used_img_id += 1
            if used_img_id == config.used_wm_len:
                break                    
            img_loop.set_postfix(used_img_id=str(used_img_id)+"/100")
            img_loop.set_description("Idx:{}, used_img_id:{}".format(b, used_img_id))
        return used_imgs, used_targets
    


    def gen_adv_lukas(watermarkloader):
        water_adv = []
        y_adv = []
        acc = 0
        num = 0

        for batch_idx, (input, label) in enumerate(tqdm(watermarkloader, unit="images", desc="Training adv exp for (watermark)"), 0):
            trigger, target = lukas(input,label , victim_model=victim_model, config=config, ref_models=ref_models, sur_models=sur_models)
            trigger, target = trigger.cuda(), target.cuda()
            pred = victim_model(trigger).argmax(axis=1)
            # print(target, pred)
            acc += sum(pred==target).item()
            num += target.size(0)
            water_adv.append(trigger.cpu().numpy())
            y_adv.append(target.cpu().numpy())

        acc = acc/num
        water_adv = np.concatenate(water_adv)
        y_adv = np.concatenate(y_adv)

        print("adv acc: ", acc)
        return water_adv, y_adv

    if args.new_lukas:
        if not os.path.exists(f"{args.save_dir}/"):
            os.makedirs(f"{args.save_dir}/")
        water_adv, y_adv = gen_adv_lukas(watermarkloader)
        torch.save(victim_model.state_dict(), f"{args.save_dir}/victim_model.pth")
        np.save(f"{args.save_dir}/adv_examples.npy", water_adv)
        np.save(f"{args.save_dir}/adv_targets.npy", y_adv)
        print("Gen adv watermark time: ", time.time()- start)
    else:
        victim_model.load_state_dict(torch.load(f"{args.save_dir}/victim_model.pth"))
        water_adv = np.load(f"{args.save_dir}/adv_examples.npy")
        y_adv = np.load(f"{args.save_dir}/adv_targets.npy")
    

    water_gen = SimpleDataset([(img,label) for img, label in zip(water_adv, y_adv)])
    watermarkloader_a = data.DataLoader(water_gen, batch_size=config.train.batch_size, shuffle=False)

    for i in range(config.num_ref_model):
        print("Test Watermark Acc on Ref Model {}: ".format(i))
        wmacc=test_watermark(ref_models[i], watermarkloader_a)
        with open(f"{args.save_dir}/result.log","a+") as file:
            file.write("refmodel{}, wmacc1: {},\n".format(i, wmacc))   

    for i in range(config.num_sur_model):
        print("Test Watermark Acc on Sur Model {}: ".format(i))
        wmacc=test_watermark(sur_models[i], watermarkloader_a)
        with open(f"{args.save_dir}/result.log","a+") as file:
            file.write("surmodel{}, wmacc1: {},\n".format(i, wmacc))   

    print("Test Watermark Acc on Victim Model: ")
    wmacc=test_watermark(victim_model, watermarkloader_a)
    with open(f"{args.save_dir}/result.log","a+") as file:
        file.write("victim_model, wmacc1: {},\n".format(wmacc))   
    # print("Test Model Acc on Victim Model: ")
    # test_watermark(victim_model, testloader)
    
    print("Test Watermark Acc on Ind Model: ")
    ind_models=load_model(config, 2)
    wmacc=test_watermark(ind_model, watermarkloader_a)
    with open(f"{args.save_dir}/result.log","a+") as file:
        file.write("ind, wmacc1: {},\n".format(wmacc))   
    for i in range(config.num_ind_model):
        wmacc=test_watermark(ind_models[i], watermarkloader_a)
        with open(f"{args.save_dir}/result.log","a+") as file:
            file.write("ind{}, wmacc1: {},\n".format(i, wmacc))   
    # print("Test Model Acc on Ind Model: ")
    # test_watermark(ind_model, testloader)
    trainloader = extract_dataset(victim_model, trainloader, config.train.batch_size, config)
    print("finetune:")
    for j in range(3):
        fint_model = config.model()
        fint_model.load_state_dict(victim_model.state_dict())
        for i in range(5):
            train_sur(config, fint_model,trainloader,n_epoch=1)
            wmacc=test_watermark(fint_model, watermarkloader_a)
            testacc=test_watermark(fint_model, testloader)
            with open(f"{args.save_dir}/result.log","a+") as file:
                file.write("fintmodel{}, epoch:{}, acc: {}, wmacc1: {},\n".format(j,i,testacc, wmacc))   
        torch.save(fint_model.state_dict(), f"{args.save_dir}/fint_model{j}.pth")
if __name__ == "__main__":
    main()