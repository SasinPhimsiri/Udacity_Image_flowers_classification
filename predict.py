import os
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
from models import load_data, save_checkpoint

parser = argparse.ArgumentParser()

parser.add_argument('image_path', default="flowers", type=float, help="Path of image to predict")
parser.add_argument('input_checkpoint', default="checkpoint_densenet161.pth",  type=str, help="Checkpoint model .pth")
parser.add_argument('--category_names',  default='cat_to_name.json', type=str, help=".json file of class names.")
parser.add_argument('--top_k', default=5, type=str, help="top K most likely classes")
parser.add_argument('--gpu', default=False, help="Select GPU")

args = parser.parse_args()

image_dir = args.image_path
model_name = args.input_checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() and gpu==True else "cpu")

checkpoint = torch.load(model_name)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    preprocessing = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                    ])
    img = Image.open(image)
    img_tensor = preprocessing(img).float().unsqueeze(0)
    img_np = np.array(img_tensor)
    return img_np

def predict(image_dir, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_dir)
    img = torch.from_numpy(img)
    img = img.to(device)
    model = model.to(device)
    pred = model(img)
    sorftmax = nn.Softmax()
    pred = sorftmax(pred)
    
    probs_top = torch.topk(pred,top_k)[0].cpu().detach().numpy()
    labels_top = torch.topk(pred,top_k)[1].cpu().detach().numpy()
    
    return img, probs_top5, labels_top

def display_predict(img_path, model, topk):
    """Display image and preditions from model"""
    # predict
    img, probs, labels = predict(img_path, model, topk)
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

if __name__ == "__main__":
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(2208, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.4),
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.4),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    display_predict(random.choice(image_dir), model, topk=top_k)