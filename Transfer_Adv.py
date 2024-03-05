import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import random_perturb, comp_prob
import os

class Transfer2():
    def __init__(self, model, ref_models, config, max_iter=100):
        super().__init__()
        self.model = model
        model.eval()
        self.ref_models = ref_models[:config.num_ref_model-3]
        for ref_model in self.ref_models:
            ref_model.eval()
        self.ind_models = ref_models[config.num_ref_model-3:]
        for ind_model in self.ind_models:
            ind_model.eval()
            
        if config.dataset.name=='CelebA':
            import boto3
            self.client = boto3.client('rekognition')
            self.celeba_temp = "exp_log/{}/celeba_temp/".format(config.imgdir)
            if not os.path.isdir(self.celeba_temp):
                os.makedirs(self.celeba_temp)
            self.n_classes=config.model.num_attributes+1
        else:
            self.n_classes=config.model.num_classes
        self.max_iter = max_iter
        self.wm_len = config.used_wm_len
        self.config = config
        if not os.path.isdir("exp_log/{}/ta_imgs".format(self.config.imgdir)):
            os.makedirs("exp_log/{}/ta_imgs/".format(self.config.imgdir))
            os.makedirs("exp_log/{}/un_imgs/".format(self.config.imgdir))

    def __call__(self, *args, **kwds):
        adv = self.forward(*args, **kwds)
        return adv

    def forward(self, images, labels, given_labels=None):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            images = images.cuda()
            self.model = self.model.cuda()
        else:
            print("Using CPU")

        criterion = nn.CrossEntropyLoss()
        targets = torch.zeros_like(labels)

        used_img_id = 0
        used_imgs = torch.zeros([self.wm_len,images.shape[1],images.shape[2],images.shape[3]])
        used_targets = torch.zeros([self.wm_len])
        if self.config.dataset.name == 'CelebA':
            Amazon_label = torch.zeros([self.wm_len])
        img_loop = tqdm(range(images.shape[0]),desc="Transfer Generating...")

        for b in img_loop:
            image = images[b:b+1,:,:,:].detach().clone()
            if self.config.dataset.name == 'CelebA' and self.config.train.name != 'DI':
                plt.imsave(self.celeba_temp+'/temp.jpg',((image-torch.min(image))/(torch.max(image)-torch.min(image))).cpu().numpy().squeeze().transpose([1,2,0]))
                with open(self.celeba_temp+'/temp.jpg','rb') as img:
                    response = self.client.detect_faces(Image={'Bytes': img.read()},    Attributes=[
                        'ALL'
                    ])
                if  not int(response['FaceDetails'][0]['Gender']['Value']=='Male') == labels[b].item():
                    continue
                # image = arctanh(image)
                image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
            if given_labels is None:
                target = labels[b] ##改成非target label
                while target == labels[b]:
                    target = torch.randint(self.n_classes, (1,))
                targets[b] = target.detach()
            else:
                target = given_labels[b].detach().cpu().long().unsqueeze(0)
                targets[b] = target.detach()
            noise = torch.zeros_like(image)
            noise.requires_grad_()
            loop = tqdm(range(self.max_iter),desc="Adv generating.", ncols=150)
            max_noise = torch.ones_like(image) * (self.config.eps/255)*(torch.max(image)-torch.min(image))
            arrive = 0
            if self.config.dataset.name != 'CIFAR10':
                arrive=1
            for i in loop:
                x = torch.clamp(image+noise, image-max_noise, image+max_noise)
                output = self.model(x)[0]

                loss = 0
                output = self.model(x)[0]
                loss_0 = criterion(output.unsqueeze(0), target.cuda())
                loss_ref = torch.zeros_like(loss_0)
                for ref_model in self.ref_models:
                    ref_out = ref_model(x)[0]
                    loss_ref = loss_ref + criterion(ref_out.unsqueeze(0), target.cuda())/len(self.ref_models)
                loss = (loss_0 + loss_ref)
                grad = torch.autograd.grad(loss, noise, retain_graph=True, create_graph=True)[0]
                noise = (noise - self.config.alpha*torch.sign(grad)).detach().cuda().requires_grad_()
                # loop.set_postfix(loss=loss.item(),label=labels[b].item(), target=target.item(),pred=output.argmax().item())
                # loop.set_description("Idx:{}, Iter:{}".format(b, i))
                loss_ind = 0
                for ind_model in self.ind_models:
                    ind_out = ind_model(x)[0]
                    loss_ind = loss_ind + criterion(ind_out.unsqueeze(0), target.cuda())/len(self.ind_models)
                loop.set_postfix(loss=loss_ind.item(),label=labels[b].item(), target=target.item(),pred=output.argmax().item())
                loop.set_description("Idx:{}, Iter:{}".format(b, i))
                if arrive==0:
                    if loss_ind < self.config.ta_loss_threshold:
                        arrive=i
                        if arrive>=10:
                            break
            
            if loss_ind > self.config.ta_loss_threshold or arrive>=10:
                continue
            # plt.imsave("exp_log/{}/ta_imgs/ta_{}_label{}-{}.jpg".format(self.config.imgdir,used_img_id,labels[b].item(),target.item()),np.concatenate([((image-torch.min(image))/(torch.max(image)-torch.min(image))).cpu().numpy().squeeze().transpose([1,2,0]),((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0])]))
            if self.config.dataset.name=='CelebA' and self.config.train.name != 'DI':
                plt.imsave(self.celeba_temp+'/temp.jpg',((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0]))
                with open(self.celeba_temp+'/temp.jpg','rb') as img:
                    response = self.client.detect_faces(Image={'Bytes': img.read()},    Attributes=[
                        'ALL'
                    ])
                    if len(response['FaceDetails'])>0:
                        Amazon_label[used_img_id] = int(response['FaceDetails'][0]['Gender']['Value']=='Male')
                    else:
                        continue

            used_imgs[used_img_id:used_img_id+1,:,:,:] = x.detach().clone()
            used_targets[used_img_id]=output.argmax().detach()
            used_img_id += 1
            if used_img_id ==self.wm_len:
                break                    
            img_loop.set_postfix(used_img_id=str(used_img_id)+"/100", \
                                l0_noise = torch.norm((x-image)/(torch.max(image)-torch.min(image)),p=float('inf')).item(),\
                                l2_noise = torch.norm((x-image)/(torch.max(image)-torch.min(image))).item())
            img_loop.set_description("Idx:{}, used_img_id:{}".format(b, used_img_id))
        if self.config.dataset.name=='CelebA' and self.config.train.name != 'DI':
            return used_imgs, used_targets.long(), Amazon_label.long()
        else:
            return used_imgs, used_targets.long()
    
    
class Transfer_Untargeted():
    def __init__(self, model, ref_models, config, max_iter=100):
        super().__init__()
        self.model = model
        self.model.eval()
        self.ref_models = ref_models
        for ref_model in self.ref_models:
            ref_model.eval()
        if config.dataset.name=='CelebA':
            import boto3
            self.client = boto3.client('rekognition')
            self.celeba_temp = "exp_log/{}/celeba_temp/".format(config.imgdir)
            if not os.path.isdir(self.celeba_temp):
                os.makedirs(self.celeba_temp)
            self.n_classes=config.model.num_attributes+1
        else:
            self.n_classes=config.model.num_classes
        self.max_iter = max_iter
        self.wm_len = config.used_wm_len
        self.config = config
        if not os.path.isdir("exp_log/{}/ta_imgs".format(self.config.imgdir)):
            os.makedirs("exp_log/{}/ta_imgs/".format(self.config.imgdir))
            os.makedirs("exp_log/{}/un_imgs/".format(self.config.imgdir))

    def __call__(self, *args, **kwds):
        adv = self.forward(*args, **kwds)
        return adv

    def forward(self, images, labels): # IFSGSM--selected
        print("current: IFGSM")
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            images = images.cuda()
            self.model = self.model.cuda()
        else:
            print("Using CPU")

        criterion = nn.CrossEntropyLoss()
        targets = torch.zeros_like(labels)

        used_img_id = 0
        used_imgs = torch.zeros([self.wm_len,images.shape[1],images.shape[2],images.shape[3]])
        used_targets = torch.zeros([self.wm_len])
        noises = torch.zeros_like(used_imgs)
        if self.config.dataset.name == 'CelebA':
            Amazon_label = torch.zeros([self.wm_len])
        img_loop = tqdm(range(images.shape[0]),desc="Transfer Generating...")
        for b in img_loop:
            image = images[b:b+1,:,:,:]
            if self.config.dataset.name == 'CelebA':
                plt.imsave(self.celeba_temp+'/temp.jpg',((image-torch.min(image))/(torch.max(image)-torch.min(image))).cpu().numpy().squeeze().transpose([1,2,0]))
                with open(self.celeba_temp+'/temp.jpg','rb') as img:
                    response = self.client.detect_faces(Image={'Bytes': img.read()},    Attributes=[
                        'ALL'
                    ])
                if  not int(response['FaceDetails'][0]['Gender']['Value']=='Male') == labels[b].item():
                    continue
                image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
            noise = torch.zeros_like(image)
            noise.requires_grad_()
            label = labels[b]
            label = label.unsqueeze(0).type(torch.long)
            loop = tqdm(range(self.max_iter),desc="Adv generating.", ncols=150)
            max_noise = torch.ones_like(image) * (self.config.eps/255)*(torch.max(image)-torch.min(image))
            for i in loop:
                # x = torch.clamp(image+noise, image-max_noise, image+max_noise)
                x = image+noise
                output = self.model(x)[0]
                # if output.argmax().item()!=label.item():
                #     x = image+noise_ls
                #     break

                loss = 0
                output = self.model(x)[0]
                # if output.shape[1]==2:
                #     output = output[0]
                loss_0 = criterion(output.unsqueeze(0), label.cuda())
                loss_ref = torch.zeros_like(loss_0)
                for ref_model in self.ref_models:
                    ref_out = ref_model(x)[0]
                    # if ref_out.shape[1]==2:
                    #     ref_out = ref_out[0]
                    loss_ref = loss_ref + criterion(ref_out.unsqueeze(0), label.cuda())/len(self.ref_models)

                loss = (loss_0 + loss_ref)
                loop.set_postfix(loss=loss.item(),target=label.item(),pred=output.argmax().item(), label=labels[b].item())
                grad = torch.autograd.grad(loss, noise, retain_graph=False, create_graph=False)[0]
                # noise_ls = noise.clone().detach()
                noise = (noise + self.config.alpha*torch.sign(grad)).detach().requires_grad_()
                # noise = torch.clamp(noise+image, 0,1)-image
                noise = torch.clamp(noise,-max_noise,max_noise)

                loop.set_description("Idx:{}, Iter:{}".format(b, i))           
            output = self.model(x)[0]
            if loss < self.config.un_loss_threshold:
                continue
            # plt.imsave("exp_log/{}/un_imgs/un_{}_label{}-{}.jpg".format(self.config.imgdir,used_img_id,label.item(),output.argmax().item()),np.concatenate([((image-torch.min(image))/(torch.max(image)-torch.min(image))).cpu().numpy().squeeze().transpose([1,2,0]),((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0])]))
            if self.config.dataset.name=='CelebA':
                plt.imsave(self.celeba_temp+'/temp.jpg',((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0]))
                with open(self.celeba_temp+'/temp.jpg','rb') as img:
                    response = self.client.detect_faces(Image={'Bytes': img.read()},    Attributes=[
                        'ALL'
                    ])
                    Amazon_label[used_img_id] = int(response['FaceDetails'][0]['Gender']['Value']=='Male')
            used_imgs[used_img_id:used_img_id+1,:,:,:] = x.detach().clone()
            noises[used_img_id:used_img_id+1,:,:,:] = (x-image).detach().clone()
            used_targets[used_img_id]=output.argmax().detach()
            used_img_id += 1
            if used_img_id ==self.wm_len:
                break                    
            img_loop.set_postfix(used_img_id=str(used_img_id)+"/100", \
                                l0_noise = torch.norm((x-image)/(torch.max(image)-torch.min(image)),p=float('inf')).item(),\
                                l2_noise = torch.norm((x-image)/(torch.max(image)-torch.min(image))).item())
            img_loop.set_description("Idx:{}, used_img_id:{}".format(b, used_img_id),)
        if self.config.dataset.name=='CelebA':
            return used_imgs, used_targets.long(), Amazon_label.long()
        else:
            return used_imgs, used_targets.long()


class Transfer2_Imagenet():
    def __init__(self, model, ref_models, config, max_iter=50):
        super().__init__()
        self.model = model
        self.model.eval()
        self.ref_models = ref_models
        for ref_model in self.ref_models:
            ref_model.eval()
        self.n_classes=config.model.num_classes
        self.max_iter = max_iter
        self.wm_len = config.used_wm_len
        self.config = config
        if not os.path.isdir("exp_log/{}/ta_imgs".format(self.config.imgdir)):
            os.makedirs("exp_log/{}/ta_imgs/".format(self.config.imgdir))
            os.makedirs("exp_log/{}/un_imgs/".format(self.config.imgdir))

    def __call__(self, *args, **kwds):
        adv = self.forward(*args, **kwds)
        return adv

    def forward(self, images, labels, given_target=None):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            images = images.cuda()
            self.model = self.model.cuda()
        else:
            print("Using CPU")

        criterion = nn.CrossEntropyLoss()
        targets = torch.zeros_like(labels)

        used_img_id = 0
        used_imgs = torch.zeros([self.wm_len,images.shape[1],images.shape[2],images.shape[3]])
        used_targets = torch.zeros([self.wm_len])

        img_loop = tqdm(range(images.shape[0]),desc="Transfer Generating...")

        for b in img_loop:
            image = images[b:b+1,:,:,:]
            scale_size = torch.max(image)-torch.min(image)
            if given_target is None:
                target = labels[b] ##change to target label
                while target == labels[b]:
                    target = torch.randint(self.n_classes, (1,))
                targets[b] = target.detach()
            else:
                target = given_target[b].detach().cpu().long().unsqueeze(0)
                targets[b] =target
            noise = torch.zeros_like(image)
            noise.requires_grad_()
            max_noise = torch.ones_like(image) * (self.config.eps/255)*(torch.max(image)-torch.min(image))

            loop = tqdm(range(self.max_iter),desc="Adv generating.", ncols=150)
            for i in loop:
                x = image+noise
                output = self.model(x)[0]

                loss = 0
                output = self.model(x)[0]
                loss_0 = criterion(output.unsqueeze(0), target.cuda())
                loss_ref = torch.zeros_like(loss_0)
                for ref_model in self.ref_models:
                    ref_out = ref_model(image + noise)[0]
                #     # loss_ref += ref_out[target]/len(self.ref_models)
                    loss_ref = loss_ref + criterion(ref_out.unsqueeze(0), target.cuda())/len(self.ref_models)
                loss = (loss_0 + loss_ref)
                loop.set_postfix(loss=loss.item(),target=target.item(),pred=output.argmax().item(), label = labels[b].item())
                grad = torch.autograd.grad(loss, noise, retain_graph=True, create_graph=True)[0]
                noise = (noise - self.config.alpha*torch.sign(grad)).detach().cuda().requires_grad_()
                noise = torch.clamp(noise, -max_noise, max_noise)
                loop.set_description("Idx:{}, Iter:{}".format(b, i))
                
            # plt.imsave("exp_log/{}/ta_imgs/ta_{}_lab{}.jpg".format(self.config.imgdir, used_img_id),np.concatenate([((image-torch.min(image))/scale_size).cpu().numpy().squeeze().transpose([1,2,0]),((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0])]))
            if loss > self.config.ta_loss_threshold:
                continue
            # plt.imsave("exp_log/{}/ta_imgs/ta_{}_lab{}.jpg".format(self.config.imgdir, used_img_id, labels[b].item()), ((image-torch.min(image))/scale_size).cpu().numpy().squeeze().transpose([1,2,0]))
            # plt.imsave("exp_log/{}/ta_imgs/ta_{}_lab{}.jpg".format(self.config.imgdir, used_img_id,target.item()),((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0]))
     
            
            used_imgs[used_img_id:used_img_id+1,:,:,:] = x.detach().clone()
            used_targets[used_img_id]=output.argmax().detach()
            used_img_id += 1
            if used_img_id ==self.wm_len:
                break                    
            img_loop.set_postfix(used_img_id=str(used_img_id)+"/100")
            img_loop.set_description("Idx:{}, used_img_id:{}".format(b, used_img_id))
            # images[b:b+1,:,:,:] = torch.clamp(image+noise.detach(), 0, 1).detach().clone()
        return used_imgs, used_targets.long()
    
    
class Transfer_Untargeted_Imagenet():
    def __init__(self, model, ref_models, config, max_iter=50):
        super().__init__()
        self.model = model
        model.eval()
        self.ref_models = ref_models
        for ref_model in ref_models:
            ref_model.eval()
        self.n_classes=config.model.num_classes
        self.max_iter = max_iter
        self.wm_len = config.used_wm_len
        self.config = config
        if not os.path.isdir("exp_log/{}/ta_imgs".format(self.config.imgdir)):
            os.makedirs("exp_log/{}/ta_imgs/".format(self.config.imgdir))
            os.makedirs("exp_log/{}/un_imgs/".format(self.config.imgdir))

    def __call__(self, *args, **kwds):
        adv = self.forward2(*args, **kwds)
        return adv

    def forward2(self, images, labels):
        print("current: forward2")
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            images = images.cuda()
            self.model = self.model.cuda()
        else:
            print("Using CPU")

        criterion = nn.CrossEntropyLoss()
        targets = torch.zeros_like(labels)

        used_img_id = 0
        used_imgs = torch.zeros([self.wm_len,images.shape[1],images.shape[2],images.shape[3]])
        used_targets = torch.zeros([self.wm_len])

        img_loop = tqdm(range(images.shape[0]),desc="Transfer Generating...")

        for b in img_loop:
            image = images[b:b+1,:,:,:]
            # image = (image-torch.min(image))/(torch.max(image)-torch.min(image))
            max_noise = torch.ones_like(image) * (self.config.eps/255)*(torch.max(image)-torch.min(image))
            noise = torch.zeros_like(image)
            noise.requires_grad_()
            label = labels[b]
            label = label.unsqueeze(0).type(torch.long)
            loop = tqdm(range(self.max_iter),desc="Adv generating.", ncols=150)
            for i in loop:
                # x = torch.clamp(image+noise, 0, 1)
                # output = self.model(x)[0]
                x = image+noise
                loss = 0
                output = self.model(x)[0]
                loss_0 = criterion(output.unsqueeze(0), label.cuda())
                loss_ref = torch.zeros_like(loss_0)
                for ref_model in self.ref_models:
                    ref_out = ref_model(x)[0]
                #     # loss_ref += ref_out[target]/len(self.ref_models)
                    loss_ref = loss_ref + criterion(ref_out.unsqueeze(0), label.cuda())/len(self.ref_models)
                loss = (loss_0 + loss_ref)
                loop.set_postfix(loss=loss.item(),target=label.item(),pred=output.argmax().item())
                grad = torch.autograd.grad(loss, noise, retain_graph=True, create_graph=True)[0]
                noise = (noise + self.config.alpha*torch.sign(grad)).detach().cuda().requires_grad_()
                noise = torch.clamp(noise,-max_noise,max_noise)
                loop.set_description("Idx:{}, Iter:{}".format(b, i))
                # plt.imsave("exp_log/{}/un_imgs/un_{}.jpg".format(self.config.imgdir, used_img_id),np.concatenate([((image-torch.min(image))/(torch.max(image)-torch.min(image))).cpu().numpy().squeeze().transpose([1,2,0]),((x-torch.min(x))/(torch.max(x)-torch.min(x))).detach().cpu().numpy().squeeze().transpose([1,2,0])]))

            # x = torch.clamp(image+noise, 0, 1)
            output = self.model(x)[0]

            if loss < self.config.un_loss_threshold:
                continue

            used_imgs[used_img_id:used_img_id+1,:,:,:] = x.detach().clone()
            used_targets[used_img_id]=output.argmax().detach()
            used_img_id += 1
            if used_img_id ==self.wm_len:
                break                    
            img_loop.set_postfix(used_img_id=str(used_img_id)+"/100")
            img_loop.set_description("Idx:{}, used_img_id:{}".format(b, used_img_id))

            # if torch.norm(grad)<1e-5:
                # break
            # plt.imsave("image.jpg",image.cpu().numpy().squeeze().transpose([1,2,0]))
            # plt.imsave( "IM+Nos.jpg",torch.clamp(image+noise.detach(), 0, 1).detach().cpu().numpy().squeeze().transpose([1,2,0]))\
            # plt.imsave("un_image.jpg",np.concatenate([image.cpu().numpy().squeeze().transpose([1,2,0]),torch.clamp(x.detach(), 0, 1).detach().cpu().numpy().squeeze().transpose([1,2,0])]))
            # images[b:b+1,:,:,:] = torch.clamp(image+noise.detach(), 0, 1).detach().clone()
            # targets[b] = output.argmax().detach()
        return used_imgs, used_targets.long()
