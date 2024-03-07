import torch
from torch import nn
from torchvision.datasets.cifar import CIFAR10,CIFAR100
from torchvision.transforms import transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import mlconfig
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
import multiprocessing
import numpy as np
import os
import mlconfig
from tqdm import tqdm
from PIL import Image
import errno
mlconfig.register(CIFAR10)


class SimpleDataset(data.Dataset):
    def __init__(self, dataset):
        self.data, self.labels = zip(*dataset)
        self.count = len(self.labels)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count

class StealDataset(data.Dataset):
    def __init__(self, dataset, model, batchsize):
        self.dataset = dataset
        self.count = dataset.__len__()
        self.model = model
        self.batchsize = batchsize
        self.__replace_labels_with_source(model)
    
    def __replace_labels_with_source(self, model):
        data_loader = torch.utils.data.DataLoader(self.dataset, 
                                                  batch_size=self.batchsize, 
                                                  shuffle=False,
                                                  num_workers=multiprocessing.cpu_count()-2,
                                                  pin_memory=True)
        self.target = torch.zeros(self.count).long()
        batch_size = self.batchsize
        model = model.cuda()
        model.eval()
        with torch.no_grad(), tqdm(data_loader, desc="Predict Stolen Labels") as pbar:
            accs = []
            for batch_id, (batch_x, y) in enumerate(pbar):
                if y.min()<0:
                    y = y.clamp_min(0).T[0]
                x = batch_x.cuda()
                output = model(x)
                if type(output)==list:
                    output = output[0]
                if type(output)==np.ndarray:
                    batch_y = torch.from_numpy(output.argmax(1))
                else:
                    batch_y = output.argmax(1)
                self.target[batch_id * batch_size:batch_id * batch_size + batch_y.shape[0]] = torch.LongTensor(batch_y.detach().cpu())
                if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100 batches.
                    accs.append(batch_y.eq(y.cuda()).cpu().sum()/batch_y.shape[0])
                    pbar.set_description(f"Stolen Labels ({100 * np.mean(accs):.4f}% Accuracy)")


    def __getitem__(self, index: int):
        return self.dataset[index][0], self.target[index]
    
    def __len__(self) -> int:
        return self.count

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

@mlconfig.register()
class CelebA(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []
        
        for idx, line in enumerate(open(os.path.join(root, ann_file), 'r')):
            if idx < 2:
                continue
            sample = line.split()
            if len(sample) != 41:
                raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append(np.clip(int(sample[21]), 0, 1))
        self.images = [os.path.join(root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
		
    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.tensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)

def steal_celeba(config, model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset= config.dataset(transform = transform)
    trainset, testset = data.random_split(train_dataset, lengths=[int(train_dataset.__len__()*0.8),train_dataset.__len__()-int(train_dataset.__len__()*0.8)], generator=torch.Generator().manual_seed(0))
    trainset = StealDataset(trainset, model)
    # _, watermarkset = data.random_split(trainset, lengths=[trainset.__len__()-config.watermark_len,config.watermark_len], generator=torch.Generator().manual_seed(0)) 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
    # watermark_loader = torch.utils.data.DataLoader(watermarkset, batch_size=config.wm_batch_size, shuffle=True)
    return train_loader

def load_data(config, shuffle=False, adv=None, seed=0):
    if config.dataset.name == 'CelebA':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            # transforms.Resize([224,192], interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        if config.train.name=='Li':
            transform = transforms.Compose([
            transforms.Resize([224,192], interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset= mlconfig.instantiate(config.dataset, transform = transform)
        trainset, testset = data.random_split(train_dataset, lengths=[int(train_dataset.__len__()*0.8),train_dataset.__len__()-int(train_dataset.__len__()*0.8)], generator=torch.Generator().manual_seed(0))
        _, watermarkset = data.random_split(trainset, lengths=[trainset.__len__()-config.watermark_len,config.watermark_len], generator=torch.Generator().manual_seed(0)) 
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True,drop_last=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True,drop_last=True)
        watermark_loader = torch.utils.data.DataLoader(watermarkset, batch_size=config.wm_batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True,drop_last=True)
        return train_loader, test_loader, watermark_loader
    elif config.dataset.name == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transform_list_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
        transform_list_test = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
        
        trainset = mlconfig.instantiate(config.dataset, train=True, transform=transform_list_train)
        testset = mlconfig.instantiate(config.dataset, train=False, transform=transform_list_test)
        # import scipy.io as sio
        # w_dataset = sio.loadmat(os.path.join(config.dataset.root, "train_32x32"))
        # x_w, y_w = np.moveaxis(w_dataset['X'], -1, 0), np.squeeze(w_dataset['y'] - 1)
        # x_w = x_w/x_w.max()
        # x_w =x_w.transpose(0,3,1,2).astype(np.float)
        # x_w = torch.FloatTensor(x_w)
        # y_w = torch.LongTensor(y_w)
        # watermark = SimpleDataset([(img,label) for img, label in zip(x_w, y_w)])
        # watermarkset ,_ = data.random_split(watermark, (config.watermark_len, x_w.shape[0]-config.watermark_len))
        # watermarkset = SimpleDataset([(img,label) for img, label in watermarkset])
        if config.train.name == "DI":
            _, watermarkset = data.random_split(trainset, lengths=[trainset.__len__()-config.watermark_len,config.watermark_len], generator=torch.Generator().manual_seed(seed))
        else:
            _, watermarkset = data.random_split(testset, lengths=[testset.__len__()-config.watermark_len,config.watermark_len], generator=torch.Generator().manual_seed(seed))

        trainloader = data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        testloader = data.DataLoader(testset, batch_size=config.train.batch_size,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        watermarkloader = data.DataLoader(watermarkset, batch_size=config.wm_batch_size,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)


        if adv == None:
            trainloader = data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        elif adv:
            trainset, _ = data.random_split(trainset, lengths=[25000,25000], generator=torch.Generator().manual_seed(seed))
            trainloader = data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
            if config.train.name == "DI":
                _, watermarkset = data.random_split(trainset, lengths=[trainset.__len__()-config.watermark_len,config.watermark_len], generator=torch.Generator().manual_seed(seed))
                watermarkloader = data.DataLoader(watermarkset, batch_size=config.wm_batch_size,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        else:
            _, trainset = data.random_split(trainset, lengths=[25000,25000], generator=torch.Generator().manual_seed(seed))
            trainloader = data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        return trainloader, testloader , watermarkloader
    else:
        root = config.dataset.root
        workers=1
        pin_memory=True
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
        ])
        )
        
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        )
        if config.train.name=="DI":
            watermarkset, _ = data.random_split(train_dataset, lengths=[config.watermark_len, len(train_dataset)-config.watermark_len],
                                            generator=torch.Generator().manual_seed(seed))
        else:
            watermarkset, _ = data.random_split(val_dataset, lengths=[config.watermark_len, len(val_dataset)-config.watermark_len],
                                                generator=torch.Generator().manual_seed(seed))
        watermarkloader = data.DataLoader(watermarkset, batch_size=config.wm_batch_size,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory
        )
    
        if adv == None:
            trainloader = data.DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        elif adv:
            trainset, _ = data.random_split(train_dataset, lengths=[len(train_dataset)//2,len(train_dataset)-len(train_dataset)//2], 
                                            generator=torch.Generator().manual_seed(seed))
            trainloader = data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        else:
            _, trainset = data.random_split(train_dataset, lengths=[len(train_dataset)//2,len(train_dataset)-len(train_dataset)//2],
                                            generator=torch.Generator().manual_seed(seed))
            trainloader = data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        return trainloader, val_loader , watermarkloader
    


def load_model(config, model_num):
    model_dir = config.modeldir + config.model.name + '/'+str(model_num) + '/'
    ref_model=[]
    for i in range(config.num_ref_model):
        model = mlconfig.instantiate(config.model)
        model.load_state_dict(torch.load(model_dir+"model{}.pth".format(i)))
        model.eval()
        ref_model.append(model.cuda())
    return ref_model


def load_sur_model(config, model_num):
    model_dir = config.modeldir + 'surrogate/' + config.model.name + '/'+str(model_num) + '/'
    ref_model=[]
    for i in range(config.num_sur_model):
        model = mlconfig.instantiate(config.model)
        model.load_state_dict(torch.load(model_dir+"model{}.pth".format(i)))
        model.eval()
        ref_model.append(model.cuda())
    return ref_model

def test(model, loader, logfile=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    model = model.cuda()
    for batch_idx, (input, label) in enumerate(tqdm(loader, desc="Test model", unit='images'), 0):
        if label.min()<0:
            label = label.clamp_min(0).T[0]
        input, label = input.cuda(), label.cuda()
        outputs = model(input)
        if type(outputs)==list:
            outputs=outputs[0]
        loss = criterion(outputs, label)

        _, predict = torch.max(outputs.data, 1)

        correct += predict.eq(label).cpu().sum()
        total += label.size(0)
        test_loss += loss.item()

    print("Test result: ")
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    if logfile is not None:
        with open(logfile, 'a') as f:
            f.write('Test results:\n')
            f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.*correct/total

def test_watermark(model, loader, Text = "Test"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    model = model.cuda()
    for batch_idx, (input, label) in enumerate(tqdm(loader, desc="Test model", unit='images'), 0):
        if label.min()<0:
            label = label.clamp_min(0).T[0]
        input, label = input.cuda(), label.cuda().type(torch.long)
        outputs = model(input)
        if type(outputs)==list:
            outputs=outputs[0]
        loss = criterion(outputs, label)

        _, predict = torch.max(outputs.data, 1)

        correct += predict.eq(label).cpu().sum()
        total += label.size(0)
        test_loss += loss.item()

    print("{} result: ".format(Text))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.*correct/total
    
def random_perturb(x):
    rnd = torch.randint(32,40,(1,)).item()
    re_im = transforms.RandomCrop((rnd,rnd),pad_if_needed=True, padding_mode="edge")(x)
    
    pad_top = torch.randint(0, 40-rnd,(1,)).item()
    pad_bottom = 40-rnd-pad_top
    pad_left = torch.randint(0,40-rnd,(1,)).item()
    pad_right = 40-rnd-pad_left

    re_im = transforms.RandomCrop((40,40),padding=(pad_top,pad_bottom,pad_left, pad_right), pad_if_needed=True)(re_im)
    re_im = transforms.Resize((32,32))(re_im)
    p = 0.5
    if torch.rand((1,))>0.5:
        return re_im
    else:
        return x
    
def arctanh(imgs):
    scaling = torch.clamp(imgs, max=1, min=-1)
    x = 0.999999 * scaling

    return 0.5*torch.log((1+x)/(1-x))

def scaler(x_atanh):
    return ((torch.tanh(x_atanh))+1) * 0.5

def comp_prob(output, target):
    output = torch.softmax(output,0)
    return output[target]/output.sum()

def extract_dataset(victim_model, trainloader, batch_size, config=None):
    if trainloader.dataset.__len__()>50000:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize([224,192], interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset= config.dataset(transform = transform)
        trainset, testset = data.random_split(train_dataset, lengths=[int(train_dataset.__len__()*0.8),train_dataset.__len__()-int(train_dataset.__len__()*0.8)], generator=torch.Generator().manual_seed(0))
        train_set = StealDataset(trainset, victim_model,config.train.batch_size)
        trainloader = data.DataLoader(train_set, batch_size=config.train.batch_size,num_workers=multiprocessing.cpu_count()-2,pin_memory=True,drop_last=True)
        return trainloader
    else:
        labels =[]
        data_x = []
        loop = tqdm(trainloader, desc='Predict labels',ncols=150)
        acc = 0
        total = 0
        victim_model = victim_model.cuda()
        victim_model.eval()
        for idx, (x,y) in enumerate(loop):
            x = x.cuda()
            lab = np.argmax(victim_model(x).detach().cpu().numpy(), axis=1)
            data_x.append(x.detach().cpu().numpy())
            labels.append(lab)
            acc += np.equal(lab,y).cpu().sum()
            total += len(lab)
            
            loop.set_postfix(ACC='Acc:{}  ({}/{})'.format(100.*(acc/total), acc, total))
        data_x = np.concatenate(data_x, axis=0)
        labels= np.concatenate(labels)
        train_set = SimpleDataset([(img, labb) for img, labb in zip(data_x, labels)])
        trainloader = data.DataLoader(train_set, batch_size=batch_size,num_workers=multiprocessing.cpu_count()-2,pin_memory=True)
        return trainloader


def train_sur(config, model, train_loader, n_epoch=None, logfile=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = mlconfig.instantiate(config.sur_optimizer, model.parameters())
    scheduler = mlconfig.instantiate(config.scheduler, optimizer)
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
        if logfile is not None:
            with open(logfile, "a+") as f:
                f.write('Epoch:%d Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))