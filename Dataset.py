import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage import io

import numpy as np
import pandas as pd

import torchvision
from torchvision import transforms


# transformation for training and validation
# we use 364 is according to ImageNet: (364 / 320) = (256 / 224)
# we use horizontal flip since maybe the lateral pictures have different views
# either from left or from right
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([364, 364]),
        transforms.RandomResizedCrop(320),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([364,364]),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class CheXpertDataset(Dataset):
    """
    Customized dataset for CheXpert dataset (https://stanfordmlgroup.github.io/competitions/chexpert/)
    """
    def __init__(self, csv_file, image_root_dir, transform=None):
        """
        Args:
            - csv_file: path to trainining and validation csv file, like "./data/CheXpert-v1.0-small/train_preprocessed.csv"
            - image_root_dir: root_dir containing image path in the csv file, like "./data/"
            - transform: transformation for each image
        """
        # for the basic ones
        self.data_frame = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.image_path = self.data_frame["Path"]
        
        # for the transformation
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        # to read in the image
        image_filename = self.image_root_dir + self.image_path[index]
        image = io.imread(image_filename, as_gray=True)
        
        # sample is a dictionary which includes the image and 14 labels
        sample = {}
        
        # since the input to pre-trained network should have 3 channels
        # we need to pad it with two repeatition
        image = np.repeat(image[None,...], 3, axis=0).transpose(1,2,0)
        
        # transform the image if transform is not None
        if self.transform:
            image = self.transform(image)
            
        # add image into the sample dictionary
        sample["image"] = image
        
        # get the label for the image
        label_col_names = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
                           "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
                           "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
        
        # to get the label for each column
        # 0 --> negative
        # 1 --> positive
        # 2 --> uncertainty (No Finding has no 2)
        for label in label_col_names:
            if self.data_frame[label][index] == 0.0:
                sample[label] = torch.LongTensor([0])
            elif self.data_frame[label][index] == 1.0:
                sample[label] = torch.LongTensor([1])
            else:
                sample[label] = torch.LongTensor([2])
        
        return sample


if __name__ == "__main__":
    
    # use the real dataloader to test
    train_loader = DataLoader(CheXpertDataset(csv_file="./data/CheXpert-v1.0-small/train_preprocessed.csv", image_root_dir="./data/", transform=train_transform), batch_size=5, shuffle=True)
    
    # to get a sample data
    for batch_idx, data in enumerate(train_loader):
        if batch_idx == 1:
            break
        batched_samples = data    

    # to print the shape for each item in batched_samples
    for key, value in batched_samples.items():
        print(key, value.shape)