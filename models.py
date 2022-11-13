import os
import json
import shutil
import sys
import time
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = { 'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406],
                                                                     [0.229,0.224,0.225])]),
                        'valid': transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(size=224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485,0.456,0.406],
                                                                         [0.229,0.224,0.225])]),
                        'test': transforms.Compose([transforms.Resize(size=255),
                                                    transforms.CenterCrop(size=224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485,0.456,0.406],
                                                                         [0.229,0.224,0.225])])
                    }

    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}


    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size =64,shuffle = True)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle = True)
    
    return trainloader, validationloader, testloader

def save_checkpoint(save_path):
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'state_dict': model.state_dict(),
                 'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, save_path)

def load_checkpoint(checkpoin_path):

    checkpoint = torch.load(checkpoin_path)
    
    if model_name == "densenet161":
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(nn.Linear(2208, hidden_layer1),
                                         nn.ReLU(),
                                         nn.Dropout(drop_out),
                                         nn.Linear(hidden_layer1, hidden_layer2),
                                         nn.ReLU(),
                                         nn.Dropout(drop_out),
                                         nn.Linear(hidden_layer2, 102),
                                         nn.LogSoftmax(dim=1))

        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
    return model

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):

    img = process_image(image_path)
    img = torch.from_numpy(img)
    img = img.to(device)
    model = model.to(device)
    pred = model(img)
    sorftmax = nn.Softmax()
    pred = sorftmax(pred)
    
    probs_top5 = torch.topk(pred,5)[0].cpu().detach().numpy()
    labels_top5 = torch.topk(pred,5)[1].cpu().detach().numpy()
    
    return img, probs_top5, labels_top5


def display_predict(img_path, model, topk_5):
    """Display image and preditions from model"""
    # predict
    img, probs, labels = predict(img_path, model, topk_5)
    classes = list({category: cat_to_name[str(category)] for category in labels[0]}.values())
    img = img.cpu().squeeze()
    # show image
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)
    ax = imshow(img, ax=ax)
    # show top 5 classes
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs[0], y=classes,color=sns.color_palette()[0]);
    plt.show();