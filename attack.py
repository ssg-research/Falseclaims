import argparse
import mlconfig
import torch
import numpy as np
from tqdm import tqdm

# import os
from utils import *
from models import *
from Transfer_Adv import Transfer2_Imagenet, Transfer2, Transfer_Untargeted,Transfer_Untargeted_Imagenet
from train import train
from utils import test
import time

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass
   
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/cifar10/train/resnet.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--ind_resume', type=t_or_f, default=True, help="Set as False to train independ model")
    parser.add_argument('--inds_resume', type=t_or_f, default=True, help="Set as False to train independ models")
    parser.add_argument('--vic_resume', type=t_or_f, default=True, help="Set as False to train victim model")
    parser.add_argument('--model_count', type=int, default=1, help='Dir id of the accuser models')
    parser.add_argument('--adv', type=t_or_f, default=True, help='True for different data, False for same data')
    parser.add_argument('--adv_target', type=t_or_f, default=False, help="targeted or untargeted")

    return parser.parse_args()



def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    print(args, config)
    if args.adv == True:
        adv = 'half'
    elif args.adv == False:
        adv = 'half'
    else:
        adv = ''

    print("Load Victim Data...")
    trainloader, testloader, watermarkloader = load_data(config, adv=True)
    print("Load Victim Data finished.")

    print("Load Ref Models...")
    ref_models = load_model(config, args.model_count)
    # for i in range(config.num_ref_model):
    for i in range(1):
        test_watermark(ref_models[i], testloader,"refmodelacc")
    print("Ref Models Loaded.")
    # torch.manual_seed(777)


    print("Initial Victim/Ind Models...")
    victim_model = mlconfig.instantiate(config.model)
    ind_models = []
    ind_model = mlconfig.instantiate(config.ind_model)
    for i in range(config.num_ind_model):
        ind_model_temp = mlconfig.instantiate(config.ind_model)
        ind_models.append(ind_model_temp)
    print("Victim/Ind Models Initialized.")

    if not os.path.exists(config.logdir+"victim") or not os.path.exists(config.modeldir+"victim"):
        os.makedirs(config.logdir+"victim/", exist_ok=True)
        os.makedirs(config.modeldir+"victim/", exist_ok=True)
        os.makedirs(config.logdir+"ind/", exist_ok=True)
        os.makedirs(config.modeldir+"ind/", exist_ok=True)


    if not args.vic_resume:
        print("Train Victim Model... ")
        train(config, victim_model, trainloader, config.logdir+"victim/{}{}_victim_model.log".format(adv,args.model_count))
        test(victim_model, testloader, config.logdir+"victim/{}{}_victim_model.log".format(adv,args.model_count))
        torch.save(victim_model.state_dict(), config.modeldir+"victim/{}{}_victim_model.pth".format(adv,args.model_count))
    else:
        print("Load Victim Models... ")
        victim_model.load_state_dict(torch.load(config.modeldir+"victim/{}{}_victim_model.pth".format(adv,args.model_count)))
        print("Victim models loaded.")


    if args.adv == False:
        adv = 'half_splitA'


    args.model_count=1
    if not args.inds_resume:
        print("Train Ind Models... ")
        ind_models = []
        for i in range(config.num_ind_model):
            ind_model_temp = mlconfig.instantiate(config.ind_model)
            ind_models.append(ind_model_temp)
        if not os.path.isdir(config.logdir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name)):
            os.makedirs(config.logdir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name))
        if not os.path.isdir(config.modeldir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name)):
            os.makedirs(config.modeldir+"ind/{}{}_{}/".format(adv, args.model_count,config.ind_model.name,config.ind_model.name))
        for i in range(config.num_ind_model):
            trainloader, _, _ = load_data(config, adv=not args.adv)

            train(config, ind_models[i], trainloader, config.logdir+"ind/{}{}_{}/{}{}_{}_ind{}.log".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i))
            test(ind_models[i], testloader, config.logdir+"ind/{}{}_{}/{}{}_{}_ind{}.log".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i))
            torch.save(ind_models[i].state_dict(), config.modeldir+"ind/{}{}_{}/{}{}_{}_ind_model{}.pth".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i))
            ind_models[i].eval()

    else:
        print("Load Ind Models... ")
        ind_models = []
        for i in range(config.num_ind_model):
            ind_model_temp = mlconfig.instantiate(config.ind_model)
            ind_models.append(ind_model_temp)
        for i in range(config.num_ind_model):
            ind_models[i].load_state_dict(torch.load(config.modeldir+"ind/{}{}_{}/{}{}_{}_ind_model{}.pth".format(adv,args.model_count,config.ind_model.name,adv,args.model_count,config.ind_model.name,i)))
            ind_models[i].eval()
            ind_models[i] = ind_models[i].cuda()
        print("Ind models loaded.")


    if not args.ind_resume:
        # torch.manual_seed(667)
        print("Load Ind Data...")        
        trainloader, _, _ = load_data(config, adv=not args.adv)
        print("Load Victim Data finished.")
        print("Train Ind Model... ")
        train(config, ind_model, trainloader, config.logdir+"ind/{}{}_{}_ind.log".format(adv, args.model_count,config.ind_model.name))
        test(ind_model, testloader, config.logdir+"ind/{}{}_{}_ind.log".format(adv,args.model_count,config.ind_model.name))
        torch.save(ind_model.state_dict(), config.modeldir+"ind/"+"{}{}_{}_ind_model.pth".format(adv,args.model_count,config.ind_model.name))
    else:
        print("Load Ind Models... ")
        ind_model.load_state_dict(torch.load(config.modeldir+"ind/"+"{}{}_{}_ind_model.pth".format(adv,args.model_count,config.ind_model.name)))
        ind_model.eval()
        ind_model = ind_model.cuda()
        print("Ind models loaded.")

    victim_model.eval()
    ind_model.eval()
    # test_watermark(victim_model, testloader)
    start = time.time()
    
    if config.dataset.name=='imagenet':
        if args.adv_target:
            attack = Transfer2_Imagenet(victim_model, ref_models, config)
        else:
            attack = Transfer_Untargeted_Imagenet(victim_model, ref_models, config)
    else:
        if args.adv_target:
            attack = Transfer2(victim_model, ref_models, config)
        else:
            attack = Transfer_Untargeted(victim_model, ref_models, config)
    water_adv = []
    y_adv = []
    acc = 0
    num = 0

    for batch_idx, (input, label) in enumerate(tqdm(watermarkloader, unit="images", desc="Training adv exp for (watermark)"), 0):
        if config.dataset.name=='CelebA':
            trigger, target, Amazon_label = attack(images=input,labels=label)
            acc += sum(target == Amazon_label).item()
        else:
            trigger, target = attack(images=input,labels=label)
            trigger, target = trigger.cuda(), target.cuda()
            pred = ref_models[0](trigger).argmax(axis=1)
        # print(target, pred)
            acc += sum(pred==target).item()
        num += target.size(0)
        water_adv.append(trigger.cpu().numpy())
        y_adv.append(target.cpu().numpy())

    acc = acc/num
    print("adv acc: ", acc)

    print("Gen adv watermark time: ", time.time()- start)
    
    water_adv = np.concatenate(water_adv)
    y_adv = np.concatenate(y_adv)
    water_gen = SimpleDataset([(img,label) for img, label in zip(water_adv, y_adv)])
    watermarkloader_a = data.DataLoader(water_gen, batch_size=config.train.batch_size, shuffle=False)

    for i in range(config.num_ref_model):
        print("Test Watermark Acc on Ref Model {}: ".format(i))
        test_watermark(ref_models[i], watermarkloader_a)

    print("Test Watermark Acc on Victim Model: ")
    test_watermark(victim_model, watermarkloader_a)
    print("Test Model Acc on Victim Model: ")
    # test_watermark(victim_model, testloader)
    
    print("Test Watermark Acc on Ind Model: ")
    test_watermark(ind_model, watermarkloader_a)
    for i in range(config.num_ind_model):
        test_watermark(ind_models[i], watermarkloader_a)
    # print("Test Model Acc on Ind Model: ")
    # test_watermark(ind_model, testloader)

if __name__ == "__main__":
    main()