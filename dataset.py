from torch.utils.data import Dataset
from PIL import Image
import json

class AirPlane100Dataset(Dataset):
    def __init__(self, airplane_img_list,transform1=None, transform2=None):

        self.airplane_img_list = airplane_img_list

        with open('mapping_dict.json','r') as f:
            self.mapping_dict = json.load(f)
        self.transform1 = transform1
        self.transform2 = transform2
        
        self.imgs   = []
        self.labels = []

        for i,full_img_path in enumerate(self.airplane_img_list):
            img  = Image.open(full_img_path)
            label = full_img_path.split('/')[-2]
            if self.transform1:
                img = self.transform1(img) #output shape=(ch,h,w)
            self.imgs.append(img)
            self.labels.append(self.mapping_dict[label])
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        img = self.imgs[idx]
        if self.transform2:
            img = self.transform2(img)
        label = self.labels[idx]
        return {'img':img, 'label':label}
    
