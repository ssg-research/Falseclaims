import argparse
import mlconfig
import torch
# mlconfig.register(torch.optim.SGD)
# mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
import numpy as np
from tqdm import tqdm

# import os
from utils import *
from models import *
from utils import test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/imagenet/train/resnet.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()
    

def train(config, model, train_loader, logfile):
    criterion = nn.CrossEntropyLoss()
    optimizer = mlconfig.instantiate(config.optimizer, model.parameters())
    scheduler = mlconfig.instantiate(config.scheduler, optimizer)

    model = model.cuda()
    model.train()
    for epoch in range(config.train.num_epoches):
        correct = 0
        total = 0
        train_loss = 0
        print('\nEpoch: %d' % epoch)
        loop = tqdm(train_loader, desc="Training reference model", unit='images')
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
        scheduler.step()
        with open(logfile, 'a') as f:
            f.write('Epoch: %d\n'%epoch)
            f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                    % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print("Train result: ")
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))



def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    print(config)

    runtime = 0
    while os.path.isdir(config.modeldir+config.model.name+'/'+str(runtime)+'/'):
        if len(os.listdir(config.modeldir+config.model.name+'/'+str(runtime)+'/')) < config.num_ref_model:
            break
        runtime += 1
    print("save dir: ", config.modeldir+config.model.name+'/'+str(runtime)+'/')
    if not os.path.isdir(config.logdir+config.model.name+'/'+str(runtime)+'/'):
        os.makedirs(config.logdir+config.model.name+'/'+str(runtime)+'/')
    if not os.path.isdir(config.modeldir+config.model.name+'/'+str(runtime)+'/'):
        os.makedirs(config.modeldir+config.model.name+'/'+str(runtime)+'/')


    logfile = config.logdir + config.model.name+'/'+str(runtime)+'/'
    modelfile = config.modeldir +config.model.name+'/'+str(runtime)+'/'

    trainloader,testloader, watermarkloader = load_data(config, adv=True, seed=2)
    reference_model=[]
    for i in range(config.num_ref_model):
        print('Train reference model:')
        reference_model.append(mlconfig.instantiate(config.model))
        train(config, reference_model[i], trainloader, logfile+"model{}.log".format(i))
        test(reference_model[i], testloader, logfile+"model{}.log".format(i))
        torch.save(reference_model[i].state_dict(), modelfile+"model{}.pth".format(i))

    #train reference models

if __name__ == "__main__":
    main()