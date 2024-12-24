import numpy as np # linear algebra
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from config import opt
import pickle

from utils import elapsed_time, count_parameters

from biggan import Generator, Discriminator
from loss import wasserstein_loss_discriminator, wasserstein_loss_generator

n_classes = opt.num_classes
device = opt.device
EMA = opt.EMA
LABEL_NOISE = opt.LABEL_NOISE
LABEL_NOISE_PROB = opt.LABEL_NOISE_PROB
TIME_LIMIT = opt.TIME_LIMIT


def generate_img(netG,fixed_noise,fixed_aux_labels=None):
    if fixed_aux_labels is not None:
        gen_image = netG(fixed_noise,fixed_aux_labels).to('cpu').clone().detach().squeeze(0)
    else:
        gen_image = netG(fixed_noise).to('cpu').clone().detach().squeeze(0)
    #denormalize
    gen_image = gen_image*0.5 + 0.5
    gen_image_numpy = gen_image.numpy().transpose(0,2,3,1)
    return gen_image_numpy

def show_generate_imgs(netG,fixed_noise,fixed_aux_labels=None, save_dir='training_log',epoch=None):
    gen_images_numpy = generate_img(netG,fixed_noise,fixed_aux_labels)

    fig = plt.figure(figsize=(25, 16))
    # display 10 images from each class
    for i, img in enumerate(gen_images_numpy):
        ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
        plt.imshow(img)
        plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.show()
    plt.close()


def cycle(iterable):
    """
    dataloaderをiteratorに変換
    :param iterable:
    :return:
    """
    while True:
        for x in iterable:
            yield x

#BigGAN
def run(lr_G=3e-4,lr_D=3e-4, beta1=0.0, beta2=0.999, nz=120, epochs=2, 
        n_ite_D=1, ema_decay_rate=0.999, show_epoch_list=None, output_freq=10, train_loader=None, start_time=None):

    #G:10M, D:8M params
    netG = Generator(n_feat=36, codes_dim=24, n_classes=n_classes).to(device) #z.shape=(*,120)
    netD = Discriminator(n_feat=42, n_classes=n_classes).to(device)
    netD.load_state_dict(torch.load('training_results_model_collapse/discriminator.pth',weights_only=True))

    if EMA:
        #EMA of G for sampling
        netG_EMA = Generator(n_feat=42, codes_dim=24, n_classes=n_classes).to(device)
        netG_EMA.load_state_dict(netG.state_dict())
        for p in netG_EMA.parameters():
            p.requires_grad = False

        
    print(count_parameters(netG))
    print(count_parameters(netD))
    
    real_label = 0.9
    fake_label = 0
    
    D_loss_list = []
    G_loss_list = []
    
    # dis_criterion = nn.BCELoss().to(device)
    dis_criterion = wasserstein_loss_discriminator().to(device)
    gen_criterion = wasserstein_loss_generator().to(device
                                                    )
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))
    
    fixed_noise = torch.randn(32, nz, 1, 1, device=device)
    #fixed_noise = fixed_noise / fixed_noise.norm(dim=1, keepdim=True)
    fixed_aux_labels     = np.random.randint(0,n_classes, 32)
    fixed_aux_labels_ohe = np.eye(n_classes)[fixed_aux_labels]
    fixed_aux_labels_ohe = torch.from_numpy(fixed_aux_labels_ohe[:,:,np.newaxis,np.newaxis])
    fixed_aux_labels_ohe = fixed_aux_labels_ohe.float().to(device, non_blocking=True)

    netG.train()
    netD.train()

    ### training here
    for epoch in range(1,epochs+1):
        if elapsed_time(start_time) > TIME_LIMIT:
            print(f'elapsed_time go beyond {TIME_LIMIT} sec')
            break
        D_running_loss = 0
        G_running_loss = 0
        for ii, data in enumerate(train_loader):
            ############################
            # (1) Update D network
            ###########################
            for _ in range(n_ite_D):
                
                if LABEL_NOISE:
                    real_label = 0.9
                    fake_label = 0
                    if np.random.random() < LABEL_NOISE_PROB:
                        real_label = 0
                        fake_label = 0.9
                    
                # train with real
                netD.zero_grad()
                real_images = data['img'].to(device, non_blocking=True) 
                real_images = real_images + 0.05 * torch.randn_like(real_images)

                batch_size  = real_images.size(0)
                dis_labels  = torch.full((batch_size, 1), real_label, device=device) #shape=(*,)
                aux_labels  = data['label'].long().to(device, non_blocking=True) #shape=(*,)
                dis_output = netD(real_images, aux_labels) #dis shape=(*,1)
                errD_real  = dis_criterion(dis_output, dis_labels)
                errD_real.backward(retain_graph=True)

                # train with fake
                noise  = torch.randn(batch_size, nz, 1, 1, device=device)
                #noise = noise / noise.norm(dim=1, keepdim=True)
                aux_labels     = np.random.randint(0,n_classes, batch_size)
                aux_labels_ohe = np.eye(n_classes)[aux_labels]
                aux_labels_ohe = torch.from_numpy(aux_labels_ohe[:,:,np.newaxis,np.newaxis])
                aux_labels_ohe = aux_labels_ohe.float().to(device, non_blocking=True)
                aux_labels = torch.from_numpy(aux_labels).long().to(device, non_blocking=True)
                
                fake = netG(noise, aux_labels_ohe) #output shape=(*,3,64,64)
                dis_labels.fill_(fake_label)
                dis_output = netD(fake.detach(),aux_labels)
                errD_fake  = dis_criterion(dis_output, dis_labels)
                errD_fake.backward(retain_graph=True)
                D_running_loss += (errD_real.item() + errD_fake.item())/len(train_loader)
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            dis_labels.fill_(real_label)  # fake labels are real for generator cost
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            aux_labels     = np.random.randint(0,n_classes, batch_size)
            aux_labels_ohe = np.eye(n_classes)[aux_labels]
            aux_labels_ohe = torch.from_numpy(aux_labels_ohe[:,:,np.newaxis,np.newaxis])
            aux_labels_ohe = aux_labels_ohe.float().to(device, non_blocking=True)
            aux_labels = torch.from_numpy(aux_labels).long().to(device, non_blocking=True)
            fake  = netG(noise, aux_labels_ohe)
            
            dis_output = netD(fake, aux_labels)
            # errG   = dis_criterion(dis_output, dis_labels)
            errG = gen_criterion(dis_output, dis_labels) # modify here
            errG.backward(retain_graph=True)
            G_running_loss += errG.item()/len(train_loader)
            optimizerG.step()
        
        if EMA:
            #update netG_EMA
            param_itr = cycle(netG.parameters())
            for i,p_EMA in enumerate(netG_EMA.parameters()):
                p = next(param_itr)
                p_EMA.data = (1-ema_decay_rate)*p_EMA.data + ema_decay_rate*p.data
                p_EMA.requires_grad = False
        
        #log
        D_loss_list.append(D_running_loss)
        G_loss_list.append(G_running_loss)
        
        #output
        if epoch % output_freq == 0:
            print('[{:d}/{:d}] D_loss = {:.3f}, G_loss = {:.3f}, elapsed_time = {:.1f} min'.format(epoch,epochs,D_running_loss,G_running_loss,elapsed_time(start_time)/60))
            
        if epoch in show_epoch_list:
            print('epoch = {}'.format(epoch))
            if not EMA:
                show_generate_imgs(netG,fixed_noise,fixed_aux_labels_ohe, epoch=epoch)
            elif EMA:
                show_generate_imgs(netG_EMA,fixed_noise,fixed_aux_labels_ohe, epoch=epoch)
            
        if epoch % 100 == 0:
            if not EMA:
                torch.save(netG.state_dict(), f'generator_epoch{epoch}.pth')
            elif EMA:
                torch.save(netG_EMA.state_dict(), f'generator_epoch{epoch}.pth')
    
    if not EMA:
        torch.save(netG.state_dict(), 'generator.pth')
    elif EMA:
        torch.save(netG_EMA.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')
    
    res = {'netG':netG,
           'netD':netD,
           'nz':nz,
           'fixed_noise':fixed_noise,
           'fixed_aux_labels_ohe':fixed_aux_labels_ohe,
           'D_loss_list':D_loss_list,
           'G_loss_list':G_loss_list,
          }
    if EMA:
        res['netG_EMA'] = netG_EMA

    current_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
    save_dict = {key:value for key,value in res.items() if key not in ['netG','netD']}
    with open(f"my_dict_{current_time}.pkl", "wb") as pickle_file:
        pickle.dump(save_dict, pickle_file)
        
    return res