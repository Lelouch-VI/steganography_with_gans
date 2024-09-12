''' LIBRARIES '''
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from PIL import Image

''' LOADING DATA
References (see these to understand the basis):
  Data Loader from the original SteganoGAN GitHub: https://github.com/DAI-Lab/SteganoGAN/blob/master/steganogan/loader.py
  IStego100k Dataset: https://arxiv.org/abs/1911.05542

So this has three components:
  Preprocessing (transformations):
    By design, all of the images in IStego100k dataset are constrained to being (a) 1024*1024 and (b) diverse in composition
    However, data processing is one of the more challenging prospects of ML, and it can lead to much worse outcomes over time
    As such, is crucial to have adequate methods of preprocessing data
    These are:
      Normalization - Altering the pixel values so that they fall between a standardized range of values (typically [-1,1]), which center the pixel values around zero
        In its normalization parameters, SteganoGAN normalizes to [.5,.5,.5]
      Random Horizontal Flip - Flips the image to diversify the dataset at random to increase diversity and avoid overfitting
      Data Augmentation and Alteration - Increasing diversity lowering chances of overfitting by altering the features of the image (for instance, changing its resolution)
      Tensor Conversion - Converts the data to a pytorch tensor, which is the data structure necessary for training a pytorch-based model

  Dataset:
    In place of a normal Dataset class, SteganoGAN utilizes a custom ImageFolder class inherited from torchvision.datasets: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
    This helps with processing image data where the images are already organized according to some kind of directory structure (e.g. the MSCOCO dataset)
    Expecting this structure as input, the loader can then create custom classes, where each subfolder is processed as a separate class.
      (since ImageFolder inherits from DatasetFolder, another pytorch class, you can also use this to create custom datasets)
    Including this allows you to limit the number of images processed to those of certain classes.

    My dataset is relatively small and thus does not need to be reduced in size or organized by directory. As a result, I did not implement an ImageFolder class.
    Instead I opted to do a more standard image processing Dataset class.

  DataLoader:
    SteganoGAN implements a custom DataLoader, also inherited from torch.utils.data.DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader  
    
    While this may be true of the original, I am striving to introduce some degree of novelty in my own implementation.
    Therefore, as it adequately fits my needs, I will be implementing my own custom DataLoader.
    However, I will be taking significant inspiration from SteganoGAN on this front, essentially ending at the same ends by different means
'''

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # resize, adjust as needed
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to SteganoGAN standard
])


# DATA SET
class MyDataset(Dataset):
  def __init__(self, cover_imgs, stego_imgs, transform=None, num_classes=2):
    if len(cover_imgs) != len(stego_imgs): # the lists should be the same length
      raise ValueError("Cover images and stego images lists must be of the same length.")

    self.cover_imgs = cover_imgs
    self.stego_imgs = stego_imgs
    self.transform = transform if transform else default_transform

  def __len__(self): # length
    return len(self.cover_imgs)

  def __getitem__(self, index): # fetch an item, return the tensor
    try:
      cover_img = Image.open(self.cover_imgs[index]).convert("RGB")
      stego_img = Image.open(self.stego_imgs[index]).convert("RGB")
    except Exception as e: # exception
      raise RuntimeError(f"Error loading image at index {index}: {e}")


    if self.transform:
      cover_img = self.transform(cover_img)
      stego_img = self.transform(stego_img)

    return cover_img, stego_img


# DATA LOADER
def create_data_loader(cover_imgs, stego_imgs, transform, shuffle=False, num_workers=4, batch_size=4):
  dataset = MyDataset(cover_imgs, stego_imgs, transform)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)