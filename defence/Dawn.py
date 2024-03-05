#key functions are copied from https://github.com/cleverhans-lab/dataset-inference into one file
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

def train_dawn(config, model, train_loader, watermark_loader,testloader=None, logfile=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)

    model = model.cuda()
    model.train()
    for epoch in range(config.train.num_epoches):
        correct = 0
        total = 0
        train_loss = 0
        print('\nEpoch: %d' % epoch)
        loop = tqdm(train_loader, desc="Training sur model", unit='images')
        for batch_idx, (input, label) in enumerate(loop, 0):
            input, label = input.cuda(), label.cuda()
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()

            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(label).cpu().sum()
            total += label.size(0)
            train_loss += loss.item()
            loop.set_postfix(loss=train_loss/(batch_idx + 1), acc=100.*(correct/total).item())

        if watermark_loader is not None and epoch%config.train.rate==0:
            if config.dataset.name=='CelebA':
                rate=3
            else:
                rate=1
            for ep_wm in range(rate):
                loop = tqdm(watermark_loader, desc="Training watermark", unit='images')

                wm_c=0
                wm_t=0
                wm_los=0
                for batch_idx, (input, label) in enumerate(loop, 0):
                    input, label = input.cuda(), label.cuda()
                    optimizer.zero_grad()
                    outputs = model(input)
                    loss = criterion(outputs, label)
                    
                    loss.backward()
                    optimizer.step()

                    _, predict = torch.max(outputs.data, 1)
                    wm_c += predict.eq(label).cpu().sum()
                    wm_t += label.size(0)
                    wm_los += loss.item()
                    loop.set_postfix(loss=wm_los/(batch_idx + 1), acc=100.*(wm_c/wm_t).item())

        scheduler.step()
        if logfile is not None:
            with open(logfile, 'a') as f:
                f.write('Epoch: %d\n'%epoch)
                f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                        % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    if testloader is not None:
        test(model, testloader)
    print("Train result: ")
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/celebA/train/resnet_dawn.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--save_dir', type=str, default="defences/celeba/dawn")   
    parser.add_argument('--new_dawn', type=bool, default=True)
    parser.add_argument('--adv', default=True)
    parser.add_argument('--model_count', type=int, default=0)

    return parser.parse_args()



def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    print(args, config)
    if args.adv == True:
        adv = 'half'
    else:
        adv = ''



    print("Initial Victim/Ind Models...")
    victim_model = config.model()
    victim_model.load_state_dict(torch.load(config.modeldir+"victim/{}{}_victim_model.pth".format(adv,args.model_count)))
    print("Victim models loaded.")
    victim_model = victim_model.cuda()
    victim_model.eval()



    print("Load Victim Data...")
    trainloader, testloader, watermarkloader = load_data(config,adv=True)
    print("Load Victim Data finished.")

    # test_watermark(victim_model, testloader)
    start = time.time()
    def dawn(images, labels, config, victim_model):
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
        
            used_imgs[used_img_id:used_img_id+1,:,:,:] = image.detach().clone()
            used_targets[used_img_id]=torch.LongTensor(target)
            used_img_id += 1
            if used_img_id == config.used_wm_len:
                break                    
            img_loop.set_postfix(used_img_id=str(used_img_id)+"/100")
            img_loop.set_description("Idx:{}, used_img_id:{}".format(b, used_img_id))
        return used_imgs, used_targets
    


    def gen_adv_dawn(watermarkloader):
        water_adv = []
        y_adv = []
        acc = 0
        num = 0

        for batch_idx, (input, label) in enumerate(tqdm(watermarkloader, unit="images", desc="Training adv exp for (watermark)"), 0):
            trigger, target = dawn(input,label , victim_model=victim_model, config=config)
            trigger, target = trigger.cuda(), target.cuda().long()

            water_adv.append(trigger.cpu().numpy())
            y_adv.append(target.cpu().numpy())


        water_adv = np.concatenate(water_adv)
        y_adv = np.concatenate(y_adv)

        print("adv acc: ", acc)
        return water_adv, y_adv

    if args.new_dawn:
        if not os.path.exists(f"{args.save_dir}/"):
            os.makedirs(f"{args.save_dir}/")
        water_adv, y_adv = gen_adv_dawn(watermarkloader)
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

    train_loader = extract_dataset(victim_model, trainloader,config.train.batch_size,config)

    for j in range(5):

        sur_model = config.model()
        train_dawn(config, sur_model, train_loader, watermarkloader_a, testloader)
        torch.save(victim_model.state_dict(), f"{args.save_dir}/sur_model{j}.pth")
        testacc=test_watermark(victim_model, testloader,"Test")
        wmacc=test_watermark(sur_model, watermarkloader_a,"Sur_WM")
        # for i in range(5):
        #     print("Epoch: ",i)
        #     testacc=train_sur(config, sur_model, train_loader, n_epoch=1)
        #     wmacc=test_watermark(sur_model, watermarkloader_a,"{}_WM".format(i))
        with open(f"{args.save_dir}/result.log","a+") as file:
            file.write("fintmodel{}, testacc:{}, wmacc1: {},\n".format(j,testacc, wmacc))   
        # torch.save(sur_model.state_dict(), f"{args.save_dir}/fint_model{j}.pth")

    print("Load Ref Models...")
    ref_models = load_model(config, args.model_count)
    print("Ref Models Loaded.")
    for i in range(config.num_ref_model):
        print("Test Watermark Acc on Ref Model {}: ".format(i))
        wmacc=test_watermark(ref_models[i], watermarkloader_a)
        with open(f"{args.save_dir}/result.log","a+") as file:
            file.write("refmodel{}, wmacc1: {},\n".format(i, wmacc))  

    print("Test Watermark Acc on Victim Model: ")
    wmacc=test_watermark(victim_model, watermarkloader_a)
    with open(f"{args.save_dir}/result.log","a+") as file:
        file.write("victim_model, wmacc1: {},\n".format(wmacc))   
    # print("Test Model Acc on Victim Model: ")
    # test_watermark(victim_model, testloader)
    
    

if __name__ == "__main__":
    main()