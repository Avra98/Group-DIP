from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
import sys
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils.denoising_utils import *
from models import *
from models.cnn import cnn
from datetime import datetime
import torch
import torch.optim
import time
#from skimage.measure import compare_psnr
from utils.inpainting_utils import * 
import pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sam import SAM

import argparse

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2, num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,ino : int =0, training_size: int = 20):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)  

    def compare_psnr(img1, img2):
        MSE = np.mean(np.abs(img1-img2)**2)
        psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
        return psnr    
    
    img_np_list=[]
    img_noisy_np_list=[]
    noisy_psnr_list=[]
    train_folder = 'result/Urban100/image_SRF_2/train'
    train_noisy_folder = 'result/Urban100/image_SRF_2/train_noisy_{}'.format(sigma)

    # os.system(f"rm -rf {train_noisy_folder}/") 
    os.makedirs(train_noisy_folder, exist_ok=True)

    print(sigma)
    for i, file_path in enumerate(glob.glob(os.path.join(train_folder, '*.png'))):

        # train on 20 images
        if i == args.training_size:
            break

        filename = os.path.splitext(os.path.basename(file_path))[0]
        imsize = -1
        img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_np = img_np[0, :, :]

        img_noisy_np = img_np + np.random.normal(scale=sigma, size=img_np.shape)
        img_noisy_np = np.clip(img_noisy_np , 0, 1).astype(np.float32)
        img_np_list.append(img_np)
        img_noisy_np_list.append(img_noisy_np)

        noisy_psnr = compare_psnr(img_np,img_noisy_np)
        noisy_psnr_list.append(noisy_psnr)
        print(f"Noisy PSNR for image {i} is '{noisy_psnr}': (max, min)=({np.max(img_np), np.min(img_np)}))")
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))
            
    print(f'Training on {len(img_noisy_np_list)} images; sigma is {sigma}; showing every {show_every} steps')
   
            
    # Modify input and output depths
    input_depth = 3    
    output_depth = 1
    weight_decay = 0.0

    # Adjust loss function
    mse = torch.nn.MSELoss().type(dtype)
    img_var_list = [np_to_torch(img_np).type(dtype) for img_np in img_np_list]
    noise_var_list = [np_to_torch(img_mask_np).type(dtype) for img_mask_np in img_noisy_np_list]

    INPUT = "noise"
        
    net_input_list = [get_noise(input_depth, INPUT, img_np.shape[0:]).type(dtype) for img_np in img_np_list]
    net = skip(
        input_depth, output_depth,
        num_channels_down = [16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_up   = [16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_skip = [0]*num_layers,
        upsample_mode='nearest',
        downsample_mode='avg',
        need1x1_up = False,
        filter_size_down=5, 
        filter_size_up=3,
        filter_skip_size = 1,
        need_sigmoid=True, 
        need_bias=True, 
        pad='reflection', 
        act_fun='LeakyReLU').type(dtype)
       
    print(f"Starting optimization with optimizer '{optim}'")
    if optim =="SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
    elif optim =="ADAM":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    elif optim =="SAM":
        base_opt = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay = weight_decay,momentum = beta) 
          
    i=0
   
    tot_loss = []
    grad_list = []
    sharp=[]
    psnr_lists = [[] for _ in range(len(img_np_list))]

    def closure_sgd(j,net_input,img_var,noise_var,m):
        ## Clean image
        img_var = np_to_torch(img_var).type(dtype)
        ## Noisy image
        noise_var = np_to_torch(noise_var).type(dtype)

        ## network pass
        out = net(net_input)
        ## compute loss of network op vs noisy image 
        total_loss = mse(out, noise_var)

        if optim=="SGD" or  optim=="ADAM":      
            optimizer.zero_grad()     
            total_loss.backward()
            optimizer.step()

        elif optim=="SAM":
            total_loss.backward() 
            optimizer.first_step(zero_grad=True)
            mse(net(net_input), noise_var).backward()
            optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            loss = mse(out.detach().cpu(), img_var.detach().cpu())
                
            out_np = out.detach().cpu().numpy()[0]
            img_np = img_var.detach().cpu().numpy()
            psnr_gt  = compare_psnr(img_np, out_np)
            psnr_lists[m].append(psnr_gt)

        return psnr_gt,out_np[0,:,:]

            
    os.makedirs('result/Urban100/out_images', exist_ok=True)
    for j in range(max_steps):
        for m in range(len(img_np_list)):
            psnr,out = closure_sgd(j, net_input_list[m], img_np_list[m], img_noisy_np_list[m], m)
            if j % show_every == 0 and j != 0: 
                print(f"At step '{j}', for image '{m}', psnr is '{psnr}'")  

                out = (out - out.min()) / (out.max() - out.min())   
                plt.imsave(f'result/Urban100/out_images/out_image_{m}_{j}.png', out, cmap='gray')

        if j % show_every == 0 and j != 0:
            print("-------------------------------------------------------------")             

        #if j%show_every==0 and j!=0:  
        #    e1,e2= get_hessian_eigenvalues(net, ind_loss, net_input_list, img_np_list, img_noisy_np_list,neigs=2)
        #    print(f"At step '{j}', e1 is '{e1}', psnr is '{psnr}'")
        #    sharp.append(e1)

    torch.cuda.empty_cache()
    print("Experiment done")  

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    model_dir = 'groupdip_models'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(net.state_dict(), f'{model_dir}/model_optim-{args.optim}_sigma-{args.sigma}_{date_time}.pth')
    print(f"Model saved as model_optim{args.optim}_sigma{args.sigma}_lr{args.lr}_{date_time}.pth")


    def plot_psnr(psnr_lists):
        filedir = f"result/Urban100/psnr_images_{args.optim}_sigma{args.sigma}_lr{args.lr}/"
        os.makedirs(filedir)
        for i, psnr_list in enumerate(psnr_lists):
            plt.figure(figsize=(10, 5))
            plt.plot(psnr_list)
            plt.title(f"Image {i}")
            plt.xlabel("Iteration")
            plt.ylabel("PSNR")
            plt.grid(True)
            plt.axhline(y=noisy_psnr_list[i])
            plt.savefig(f"{filedir}/psnr_image_{i}.png") 

    plot_psnr(psnr_lists)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")
    
    parser.add_argument("--images", type=str, default = ["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=0.08, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=20000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="SAM", help="which optimizer")
    #parser.add_argument("--IGR", type=str, default="Normal", help="true if SAM ")
    parser.add_argument("--reg", type=float, default=0.05, help="if regularization strength of igr")
    parser.add_argument("--sigma", type=float, default=0.05, help="noise-level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=100, help="show_every")
    parser.add_argument("--device_id", type=int, default=0, help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0, help="momentum for sgd ")
    parser.add_argument("--ino", type=int, default=0, help="image index ")
    parser.add_argument("--training_size", type=int, default=20, help="how many images to train on")
    args = parser.parse_args()
    
    main(images=args.images, 
         lr=args.lr, 
         max_steps=args.max_steps, 
         optim= args.optim,
         reg=args.reg,sigma = args.sigma,
         num_layers = args.num_layers, 
         show_every = args.show_every,
         beta = args.beta, 
         device_id = args.device_id,
         ino = args.ino,
         training_size = args.training_size)