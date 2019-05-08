'''
This script contains a function to produce the Class Activation Maps. 
The Code is addapted from https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py
'''

#import the necessary modules

#import sys
#sys.path.insert(0,"/work/pip")
import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
import itertools
import os
import random

from utils.fs_utils import *
from models.densenet121 import make_model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

InvNormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class generateCAM():
    '''
    Class initialized with a trained model (contains final global avg. pooling layer). 
    Generates CAM from data. 
    '''
    def __init__(self, model_path, finalconv_name = 'features', Dataset = None, result_dir = './CAM'):
        #Assign model and Dataset
        model = make_model(pretrained='pretrained', fixed=False)
        model.load_state_dict(torch.load(os.path.join('./trained_models', model_path), map_location='cpu'))
        model.device = device
        model.eval()
        self.model = model
        self.Dataset = Dataset
        self.data_size = len(Dataset)
        
        #Register Forward Hook
        self.features_blobs = []
        def hook_feature(module, input, output):
            self.features_blobs.append(output.data.cpu().numpy())
        
        self.model._modules.get(finalconv_name).register_forward_hook(hook_feature)
        
        #Get the softmax weight
        params = list(self.model.parameters())
        self.weight_softmax = np.squeeze(params[-2].data.numpy())
        
        #Set/create results dir
        self.result_dir = result_dir
        create_folder(self.result_dir)
        
    def getCAM(self):
        found = False
        while not found:
            #Get Random Image from Dataset
            rand_idx = random.randint(0, self.data_size)
            print(rand_idx)
            data = self.Dataset[rand_idx]
            img_name = self.Dataset.image_path[rand_idx].replace('/', '_')
            img_tensor = data['image'].unsqueeze(0)
            data.pop('image')
            img_labels = data
            img_variable = Variable(img_tensor)
            logit = self.model(img_variable)

            #Generate Classes List
            labels = list(itertools.product([label[0] for label in img_labels.items()], [0, 1, 2]))
            labels.pop(2)
            classes = {num:label for (label, num) in zip(labels, np.arange(41))}

            #Get Postive Class
            img_labels = {k:int(v) for k, v in img_labels.items()}
            pos_labels = np.array(list(img_labels.keys()))[np.array(list(img_labels.values())) == 1].tolist()
            pos_idx = []
            for label in pos_labels:
                i = list(classes.keys())[list(classes.values()).index((label, 1))]
                pos_idx.append(i)

            #Get Model Predictions
            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs = probs.numpy()
            idx = idx.numpy()
            
            #Get Intersection Between Positive Labels and top 5 Predictions
            idx_agree = set(idx.tolist()).intersection(pos_idx)
            if bool(idx_agree):
                found = True 
                idx_sel = list(idx_agree)[0]

            #Check if top1 Predictions is in Postive Label Set
            #correct = pos_idx.count(idx) == 1

        #Generate class activation mapping for the top1 prediction
        CAMs = returnCAM(self.features_blobs[0], self.weight_softmax, [idx_sel])
        self.features_blobs = []
        
        #Generate Heatmaped Image
        img_untransformed = InvNormalize(img_tensor.squeeze(0))
        img = img_untransformed.numpy().transpose(1,2,0) * 255
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.7
        
        #Save Figure
        fig, ax = plt.subplots(1,1)
        ax.imshow(result.astype(int))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.annotate('Prediction: {}: {} \n Labels: {}'.format(classes[idx_sel][0], classes[idx_sel][1], ', '.join(pos_labels)),
            xy=(0.5, -.02), xytext=(0, 10),
            xycoords=('axes fraction', 'figure fraction'),
            textcoords='offset points',
            size=14, ha='center', va='bottom')
        fig.savefig(os.path.join(self.result_dir, '{}.png'.format(img_name)))