import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # list all the image and mask files in the data directory
        self.image_files = sorted(os.listdir(os.path.join(data_dir, "image")))
        self.mask_files = sorted(os.listdir(os.path.join(data_dir, "label")))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # read the image and mask file
        image = Image.open(os.path.join(self.data_dir, "image", self.image_files[idx]))
        mask = Image.open(os.path.join(self.data_dir, "label", self.mask_files[idx]))
        
        # apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        # convert image and mask to tensors
        image = image.clone().detach().float()
        mask = mask.clone().detach().float()
        
        # return image and mask as a tuple
        return image, mask