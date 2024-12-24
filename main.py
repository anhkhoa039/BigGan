from train import run
import time
import os
import random
import torch
import glob, json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms


from dataset import AirPlane100Dataset
from config import opt
from datetime import datetime
start_time = time.time()

seed = 2019
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pickle

def save_res(res, filename="res.pkl"):
    """
    Save the dictionary containing the models and variables to a file.
    
    Args:
        res (dict): Dictionary to save.
        filename (str): Name of the file to save the dictionary.
    """
    with open(filename, "wb") as f:
        pickle.dump(res, f)
    print(f"Dictionary saved to {filename}")


if __name__ == "__main__":
    start_time = time.time()
    img_height,img_width = opt.img_height, opt.img_width
    batch_size = opt.batch_size
    NUM_WORKERS = opt.NUM_WORKERS
    img_size    = 64
    EPOCHS = opt.EPOCHS
    MEAN1,MEAN2,MEAN3 = 0.5, 0.5, 0.5
    STD1,STD2,STD3    = 0.5, 0.5, 0.5

    data_root = opt.data_dir
    img_list_ = [glob.glob(f"{data_root}/{i}/*")[:100] if len(glob.glob(f"{data_root}/{i}/*")) >100 else glob.glob(f"{data_root}/{i}/*") for i in os.listdir(data_root) ]
    img_list = sum(img_list_,[])
    
    transform1 = transforms.Compose([transforms.Resize((img_height,img_width))])

    transform2 = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[MEAN1, MEAN2, MEAN3],
                                                        std=[STD1, STD2, STD3]),
                                    ])

    train_set = AirPlane100Dataset(
                        airplane_img_list=img_list,
                        transform1=transform1,
                        transform2=transform2,
                        )

    train_loader = DataLoader(train_set,
                            shuffle=True, batch_size=batch_size,
                            num_workers=NUM_WORKERS, pin_memory=True)


    show_epoch_list = np.arange(0,EPOCHS+10,10)
    lr_G = 3e-4 # Generator
    lr_D = 3e-4  # Discriminator
    
    res = run(lr_G=lr_G,lr_D=lr_D, beta1=0.0, beta2=0.999, nz=120, epochs=EPOCHS, 
          n_ite_D=1, ema_decay_rate=None, show_epoch_list=show_epoch_list, output_freq=10, train_loader=train_loader, start_time=start_time)
    print ("Training Done!!!, Saving model ....")
    save_res(res, filename="res.pkl")
    print ('Done!!!')

 