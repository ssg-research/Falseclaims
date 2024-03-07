import argparse
import mlconfig
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import sys 
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
# import os
from utils import *
from models import *
from train import train
# from train import train
# from utils import test
from tqdm import tqdm

from models.Discriminator import DiscriminatorNet, DiscriminatorNet_CelebA, DiscriminatorNet_ImageNet
from models.HidingUNet import UnetGenerator

import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/cifar10/train/resnet_li.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--save_dir', type=str, default='defences/cifar10/Li')    
    parser.add_argument('--ind_resume', type=bool, default=False)
    parser.add_argument('--fine_tune_resume', type=bool, default=False)
    parser.add_argument('--vic_resume', type=bool, default=False)
    parser.add_argument('--sur_resume', type=bool, default=True)
    parser.add_argument('--adv', default=True)
    parser.add_argument('--model_count', type=int, default=0)
    parser.add_argument('--hyper-parameters',  nargs='+', default=[3, 5, 1, 0.1])

    return parser.parse_args()


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, config, Dnnet, Hidnet, Disnet, trainloader,testloader, secret_img, wm_inputs, wm_labels, wm_cover_labels, wm_idx=0):
    criterionH_mse = nn.MSELoss()
    criterionH_ssim = SSIM()
    optimizerH = optim.Adam(Hidnet.parameters(), lr=0.001, betas=(0.5, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

    criterionD = nn.BCELoss()
    optimizerD = optim.Adam(Disnet.parameters(), lr=0.001, betas=(0.5, 0.999))
    schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=8, verbose=True)

    criterionN = nn.CrossEntropyLoss()
    optimizerN = optim.SGD(Dnnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedulerN = MultiStepLR(optimizerN, milestones=[40, 80], gamma=0.1)
    
    valid = torch.cuda.FloatTensor(config.wm_batch_size, 1).fill_(1.0)
    fake = torch.cuda.FloatTensor(config.wm_batch_size, 1).fill_(0.0)

    args.wm_batchsize = config.wm_batch_size
    args.batchsize = config.train.batch_size

    train_loss, test_loss = [[], []], [[], []]
    train_acc, test_acc = [[], []], [[], []]
    for epoch in range(config.train.num_epoches):
        print('\nEpoch: %d' % epoch)
        Dnnet.train()
        Hidnet.train()
        Disnet.train()
        wm_cover_correct, wm_correct, real_correct, wm_total, real_total = 0, 0, 0, 0, 0
        loss_H_ = AverageMeter()
        loss_D_ = AverageMeter()
        real_acc = AverageMeter()
        wm_acc = AverageMeter()
        loop = tqdm(trainloader)
        for batch_idx, (input, label) in enumerate(loop):
            input, label = input.cuda(), label.cuda()
            wm_input = wm_inputs[(wm_idx + batch_idx) % len(wm_inputs)]
            wm_label = wm_labels[(wm_idx + batch_idx) % len(wm_inputs)]
            wm_cover_label = wm_cover_labels[(wm_idx + batch_idx) % len(wm_inputs)]
            #############Discriminator##############
            optimizerD.zero_grad()
            wm_img = Hidnet(wm_input, secret_img)
            wm_dis_output = Disnet(wm_img.detach())
            real_dis_output = Disnet(wm_input)
            loss_D_wm = criterionD(wm_dis_output, fake)
            loss_D_real = criterionD(real_dis_output, valid)
            loss_D = loss_D_wm + loss_D_real
            loss_D.backward()
            optimizerD.step()
            ################Hidding Net#############
            optimizerH.zero_grad()
            optimizerD.zero_grad()
            optimizerN.zero_grad()
            wm_dis_output = Disnet(wm_img)
            wm_dnn_output = Dnnet(wm_img)
            loss_mse = criterionH_mse(wm_input, wm_img)
            loss_ssim = criterionH_ssim(wm_input, wm_img)
            loss_adv = criterionD(wm_dis_output, valid)

            loss_dnn = criterionN(wm_dnn_output, wm_label)

            loss_H = args.hyper_parameters[0] * loss_mse + args.hyper_parameters[1] * (1-loss_ssim) + args.hyper_parameters[2] * loss_adv + args.hyper_parameters[3] * loss_dnn
            loss_H.backward()
            optimizerH.step()
            ################DNNet#############
            optimizerN.zero_grad()
            inputs = torch.cat([input, wm_img.detach()], dim=0)
            labels = torch.cat([label, wm_label], dim=0)
            dnn_output = Dnnet(inputs)
        
            loss_DNN = criterionN(dnn_output, labels)
            loss_DNN.backward()
            optimizerN.step()

            # calculate the accuracy
            wm_cover_output = Dnnet(wm_input)
            _, wm_cover_predicted = wm_cover_output.max(1)
            wm_cover_correct += wm_cover_predicted.eq(wm_cover_label).sum().item()

            _, wm_predicted = dnn_output[args.batchsize: args.batchsize +
                                        args.wm_batchsize].max(1)
            wm_correct += wm_predicted.eq(wm_label).sum().item()
            wm_total += args.wm_batchsize

            _, real_predicted = dnn_output[0:args.batchsize].max(1)
            real_correct += real_predicted.eq(
                labels[0:args.batchsize]).sum().item()
            real_total += args.batchsize
            loop.set_postfix(real_acc=100. * real_correct / real_total, wm_acc=100. * wm_correct / wm_total, loss=loss_DNN.item())
            # print('[%d/%d][%d/%d]  Loss D: %.4f Loss_H: %.4f (mse: %.4f ssim: %.4f adv: %.4f)  Loss_real_DNN: %.4f Real acc: %.3f  wm acc: %.3f' % (
            #     epoch, config.train.num_epoches, batch_idx, len(trainloader),
            #     loss_D.item(), loss_H.item(), loss_mse.item(
            #     ), loss_ssim.item(), loss_adv.item(), loss_DNN.item(),
            #     100. * real_correct / real_total, 100. * wm_correct / wm_total))

            loss_H_.update(loss_H.item(), int(input.size()[0]))
            loss_D_.update(loss_D.item(), int(input.size()[0]))
            real_acc.update(100. * real_correct / real_total)
            wm_acc.update(100. * wm_correct / wm_total)
        train_loss[0].append(loss_H_.avg)
        train_loss[1].append(loss_D_.avg)
        train_acc[0].append(real_acc.avg)
        train_acc[1].append(wm_acc.avg)
        val_hloss, val_disloss, val_dnnloss, acc, wm_acc, wm_inut_acc = test(args, config,epoch, Dnnet, Hidnet, Disnet, testloader, wm_inputs,wm_labels, wm_idx, wm_cover_labels, secret_img)
        torch.save(wm_inputs,f"{args.save_dir}/wm_inputs")
        torch.save(wm_labels,f"{args.save_dir}/wm_labels")
        torch.save(wm_cover_labels,f"{args.save_dir}/wm_cover_labels")
        torch.save(secret_img, f"{args.save_dir}/secret_img")
        schedulerH.step(val_hloss)
        schedulerD.step(val_disloss)
        schedulerN.step()

def test(args, config,epoch, Dnnet, Hidnet,Disnet, testloader, wm_inputs, wm_labels, wm_idx, wm_cover_labels,secret_img, save=True):
    args.wm_batchsize = config.wm_batch_size
    args.batchsize = config.train.batch_size
    train_loss, test_loss = [[], []], [[], []]
    train_acc, test_acc = [[], []], [[], []]
    Dnnet = Dnnet.cuda()
    Dnnet.eval()
    Hidnet.eval()
    Disnet.eval()
    wm_cover_correct, wm_correct, real_correct, real_total, wm_total = 0, 0, 0, 0, 0
    Hlosses = AverageMeter()  
    Dislosses = AverageMeter()  
    real_acc = AverageMeter()
    wm_acc = AverageMeter()
    DNNlosses = AverageMeter()
    valid = torch.cuda.FloatTensor(config.wm_batch_size, 1).fill_(1.0)
    fake = torch.cuda.FloatTensor(config.wm_batch_size, 1).fill_(0.0)

    criterionH_mse = nn.MSELoss()
    criterionH_ssim = SSIM()
    optimizerH = optim.Adam(Hidnet.parameters(), lr=0.001, betas=(0.5, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

    criterionD = nn.BCELoss()
    optimizerD = optim.Adam(Disnet.parameters(), lr=0.001, betas=(0.5, 0.999))
    schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=8, verbose=True)

    criterionN = nn.CrossEntropyLoss()
    optimizerN = optim.SGD(Dnnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedulerN = MultiStepLR(optimizerN, milestones=[40, 80], gamma=0.1)

    watermark = []
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(testloader):
            input, label = input.cuda(), label.cuda()
            wm_input = wm_inputs[(wm_idx + batch_idx) % len(wm_inputs)]
            wm_label = wm_labels[(wm_idx + batch_idx) % len(wm_inputs)]
            wm_cover_label = wm_cover_labels[(
                wm_idx + batch_idx) % len(wm_inputs)]
            #############Discriminator###############
            wm_img = Hidnet(wm_input, secret_img)
            wm_dis_output = Disnet(wm_img.detach())
            real_dis_output = Disnet(wm_input)
            loss_D_wm = criterionD(wm_dis_output, fake)
            loss_D_real = criterionD(real_dis_output, valid)
            loss_D = loss_D_wm + loss_D_real
            Dislosses.update(loss_D.item(), int(wm_input.size()[0]))

            ################Hidding Net#############
            wm_dnn_outputs = Dnnet(wm_img)
            loss_mse = criterionH_mse(wm_input, wm_img)
            loss_ssim = criterionH_ssim(wm_input, wm_img)
            loss_adv = criterionD(wm_dis_output, valid)
         
            loss_dnn = criterionN(wm_dnn_outputs, wm_label)
            loss_H = args.hyper_parameters[0] * loss_mse + args.hyper_parameters[1] * (1-loss_ssim) + args.hyper_parameters[2] * loss_adv + args.hyper_parameters[3] * loss_dnn
            Hlosses.update(loss_H.item(), int(input.size()[0]))
            ################DNNet#############
            inputs = torch.cat([input, wm_img.detach()], dim=0)
            labels = torch.cat([label, wm_label], dim=0)
            dnn_outputs = Dnnet(inputs)
        
            loss_DNN = criterionN(dnn_outputs, labels)
            DNNlosses.update(loss_DNN.item(), int(inputs.size()[0]))

           
            wm_cover_output = Dnnet(wm_input)
            _, wm_cover_predicted = wm_cover_output.max(1)
            wm_cover_correct += wm_cover_predicted.eq(
                wm_cover_label).sum().item()

            #wm_dnn_output = Dnnet(wm_img)
            # _, wm_predicted = wm_dnn_output.max(1)
            _, wm_predicted = dnn_outputs[args.batchsize:
                                          args.batchsize + args.wm_batchsize].max(1)
            wm_correct += wm_predicted.eq(wm_label).sum().item()
            wm_total += args.wm_batchsize

            _, real_predicted = dnn_outputs[0:args.batchsize].max(1)
            real_correct += real_predicted.eq(
                labels[0:args.batchsize]).sum().item()
            real_total += args.batchsize

    val_hloss = Hlosses.avg
    val_disloss = Dislosses.avg
    val_dnnloss = DNNlosses.avg
    real_acc.update(100. * real_correct / real_total)
    wm_acc.update(100. * wm_correct / wm_total)
    test_acc[0].append(real_acc.avg)
    test_acc[1].append(wm_acc.avg)
    print('Real acc: %.3f  wm acc: %.3f wm cover acc: %.3f ' % (
        100. * real_correct / real_total, 100. * wm_correct / wm_total, 100. * wm_cover_correct / wm_total))

    resultImg = torch.cat([wm_input, wm_img, secret_img], 0)
    torchvision.utils.save_image(resultImg, 'defences/Li/' + 'images/Epoch_' + str(epoch) + '_img.png', nrow=args.wm_batchsize,
                                 padding=1, normalize=True)
    test_loss[0].append(val_hloss)
    test_loss[1].append(val_disloss)

    real_acc = 100. * real_correct / real_total
    wm_acc = 100. * wm_correct / wm_total
    wm_input_acc = 100. * wm_cover_correct / wm_total
    return val_hloss, val_disloss, val_dnnloss, real_acc, wm_acc, wm_input_acc

def train_sur(config, model, train_loader, n_epoch=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = config.sur_optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    correct = 0
    total = 0
    train_loss = 0
    model = model.cuda()
    model.train()
    if n_epoch == None:
        n_epoch = config.train.num_epoches
    for epoch in range(n_epoch):
        print('\nEpoch: %d' % epoch)
        for batch_idx, (input, label) in enumerate(tqdm(train_loader, desc="Training reference model", unit='images'), 0):
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
        scheduler.step()
        print("Train result: ")
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
        

def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    print(args, config)
    if args.adv == True:
        adv = 'half'
    else:
        adv = ''

    args.save_dir = f'defences/{config.dataset.name.lower()}/Li'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   
    print("Load Victim Data...")
    trainloader, testloader, watermarkloader = load_data(config, adv=True)
    print("Load Victim Data finished.")

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if config.dataset.name == "CelebA":
        transform_list_test = transforms.Compose([
                                transforms.Resize([224,192], interpolation=Image.BICUBIC),
                                transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
    elif config.dataset.name == "CIFAR10":
        transform_list_test = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
    else:
        transform_list_test = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        
    ieee_logo = torchvision.datasets.ImageFolder(
        root='data/datasets/IEEE', transform=transform_list_test)
    ieee_loader = torch.utils.data.DataLoader(ieee_logo, batch_size=1)
    for _, (logo, __) in enumerate(ieee_loader):
        secret_img = logo.expand(
            config.wm_batch_size, logo.shape[1], logo.shape[2], logo.shape[3]).cuda()

    wm_inputs, wm_cover_labels = [], []
    wm_labels = []
    for wm_idx, (wm_input, wm_cover_label) in enumerate(watermarkloader):
        wm_input, wm_cover_label = wm_input.cuda(), wm_cover_label.cuda()

        wm_inputs.append(wm_input)
        wm_cover_labels.append(wm_cover_label)
        if config.dataset.name == "CelebA":
            wm_labels.append(1-wm_cover_label)
    if config.dataset.name != "CelebA":
        np_labels = np.random.randint(
            2, size=(int(config.watermark_len/config.wm_batch_size), config.wm_batch_size))
        wm_labels = torch.from_numpy(np_labels).cuda()
    Dis = {'CIFAR10':DiscriminatorNet,'CelebA':DiscriminatorNet_CelebA,'imagenet':DiscriminatorNet_ImageNet}
    Hidnet = UnetGenerator()
    Hidnet = Hidnet.cuda()
    Disnet = Dis[config.dataset.name]()
    Disnet = Disnet.cuda()
    Dnnet = mlconfig.instantiate(config.model)
    Dnnet = Dnnet.cuda()
    
    global best_real_acc
    global best_wm_acc
    global best_wm_input_acc
    best_real_acc, best_wm_acc, best_wm_input_acc = 0, 0, 0
    if not os.path.exists(f"{args.save_dir}"):
        os.makedirs(f"{args.save_dir}/checkpoint")
        os.makedirs(f"{args.save_dir}/images")
    if not os.path.exists(f"{args.save_dir}/surrogate"):
        os.makedirs(f"{args.save_dir}/surrogate")
        os.makedirs(f"{args.save_dir}/independent")
    if args.vic_resume:
        print("Load wm model and watermarks")
        Dnnet.load_state_dict(torch.load(f"{args.save_dir}/Dnnet.pth"))
        Hidnet.load_state_dict(torch.load(f"{args.save_dir}/Hidnet.pth"))
        Disnet.load_state_dict(torch.load(f"{args.save_dir}/Disnet.pth"))
        wm_inputs=torch.load(f"{args.save_dir}/wm_inputs")   
        wm_labels=torch.load(f"{args.save_dir}/wm_labels")
        wm_cover_labels=torch.load(f"{args.save_dir}/wm_cover_labels")
        secret_img=torch.load(f"{args.save_dir}/secret_img")
        test(args, config, 0, Dnnet, Hidnet, Disnet, testloader, wm_inputs,wm_labels, wm_idx, wm_cover_labels, secret_img)
        wm_img = []
        wm_lab =[]
        for i in range(len(wm_inputs)):
            wm_img.append(Hidnet(wm_inputs[i], secret_img).detach().cpu().numpy())
            wm_lab.append(wm_labels[i].cpu().numpy())
        wm_img = np.concatenate(wm_img)
        wm_lab = np.concatenate(wm_lab)
        water_gen = SimpleDataset([(img,label) for img, label in zip(wm_img, wm_lab)])
        watermarkloader_a = data.DataLoader(water_gen, batch_size=config.train.batch_size, shuffle=False)
        dnn_wmacc=test_watermark(Dnnet, watermarkloader_a)
    else:
        train(args, config, Dnnet, Hidnet, Disnet, trainloader, testloader, secret_img, wm_inputs, wm_labels, wm_cover_labels, wm_idx)
        
        torch.save(Dnnet.state_dict(), f"{args.save_dir}/Dnnet.pth")
        torch.save(Hidnet.state_dict(), f"{args.save_dir}/Hidnet.pth")
        torch.save(Disnet.state_dict(), f"{args.save_dir}/Disnet.pth")
        wm_img = []
        wm_lab =[]
        for i in range(len(wm_inputs)):
            wm_img.append(Hidnet(wm_inputs[i], secret_img).detach().cpu().numpy())
            wm_lab.append(wm_labels[i].cpu().numpy())
        wm_img = np.concatenate(wm_img)
        wm_lab = np.concatenate(wm_lab)
        water_gen = SimpleDataset([(img,label) for img, label in zip(wm_img, wm_lab)])
        watermarkloader_a = data.DataLoader(water_gen, batch_size=config.train.batch_size, shuffle=False)
        dnn_wmacc=test_watermark(Dnnet, watermarkloader_a)
    with open(f"{args.save_dir}/result.log","a+") as file:
            file.write("dnn wmacc: {}, ".format(dnn_wmacc))

    with open(f"{args.save_dir}/result.log","a+") as file:
        file.write("\n")
    trainloader = extract_dataset(Dnnet, trainloader, config.train.batch_size, config)
    if args.fine_tune_resume:
        print("Load Fine-Tune models")
        for i in range(config.num_fint_model):
            fint_model = config.ind_model()
            fint_model.load_state_dict(torch.load(f"{args.save_dir}/surrogate/fint_model{i}.pth"))
            print("Test Acc:")
            _, _, _, fint_acc, fint_wmacc1, _ = test(args, config, 0, fint_model, Hidnet, Disnet, testloader, wm_inputs,wm_labels, wm_idx, wm_cover_labels, secret_img)
            print("WM Acc:")
            fint_wmacc2 = test_watermark(fint_model, watermarkloader_a)
            with open(f"{args.save_dir}/result.log","a+") as file:
                file.write("fintune{} acc: {}, wmacc1: {},wmacc2: {}, \n".format(i,fint_acc, fint_wmacc1, fint_wmacc2))           
    else:
        print("Train Fine-Tune models")
        for i in range(config.num_fint_model):
            fint_model = config.ind_model()
            fint_model.load_state_dict(torch.load(f"{args.save_dir}/Dnnet.pth"))
            test(args, config, 0, fint_model, Hidnet, Disnet, testloader, wm_inputs,wm_labels, wm_idx, wm_cover_labels, secret_img)
            print("Test Acc:")
            fint_acc = test_watermark(fint_model, testloader)
            print("WM Acc:")
            fint_wmacc = test_watermark(fint_model, watermarkloader_a)
            with open(f"{args.save_dir}/result.log","a+") as file:
                file.write("before fintune{} acc: {}, wmacc: {}, ".format(i,fint_acc, fint_wmacc))    
            # trainloader = load_finetune_data(config)
            for epoch in range(5):
                train_sur(config, fint_model, trainloader,n_epoch=1)
                # test(args, config, 0, fint_model, Hidnet, Disnet, testloader, wm_inputs,wm_labels, wm_idx, wm_cover_labels, secret_img)
                print("Test Acc:")
                t = test_watermark(fint_model, testloader)
                print("WM Acc:")
                r = test_watermark(fint_model, watermarkloader_a)
                with open(f"{args.save_dir}/result.log","a+") as file:
                    file.write("finetune{}, epoch {}, testacc: {}, wmacc: {}, \n".format(i,epoch,t,r))
            torch.save(fint_model.state_dict(), f"{args.save_dir}/surrogate/fint_model{i}.pth")    

    if args.ind_resume:
        print("Load Ref Models...")
        ref_models = load_model(config, args.model_count)
        print("Ref Models Loaded.")
        for i in range(len(ref_models)):
            print("Test Watermark Acc on Ref Model {}: ".format(i))
            _, _, _, fint_acc, fint_wmacc1, _ = test(args, config, 0, ref_models[i], Hidnet, Disnet, testloader, wm_inputs,wm_labels, wm_idx, wm_cover_labels, secret_img)   
            with open(f"{args.save_dir}/result.log","a+") as file:
                file.write("ind{} acc: {}, wmacc1: {},\n".format(i,fint_acc, fint_wmacc1))       
    else:
        print("Load Ref Models...")
        ref_models = load_model(config, args.model_count)
        print("Ref Models Loaded.")
        for i in range(len(ref_models)):
            print("Test Watermark Acc on Ref Model {}: ".format(i))
            _, _, _, fint_acc, fint_wmacc1, _  = test(args, config, 0, ref_models[i], Hidnet, Disnet, testloader, wm_inputs,wm_labels, wm_idx, wm_cover_labels, secret_img)       
            torch.save(ref_models[i].state_dict(), f"{args.save_dir}/independent/ind_model{i}.pth")
            with open(f"{args.save_dir}/result.log","a+") as file:
                file.write("ind{} acc: {}, wmacc1: {},\n".format(i,fint_acc, fint_wmacc1))   
    with open(f"{args.save_dir}/result.log","a+") as file:
        file.write("\n")  

if __name__ == "__main__":
    main()