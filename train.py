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

parser.add_argument('data_directory', default="flowers",  type=str, help="Folder dataset directory")
parser.add_argument('--save_dir',  default=os.getcwd(), type=str, help="Save checkpoint directory")
parser.add_argument('--arch', default="densenet161", type=str, help="densenet161 architecture")
parser.add_argument('--learning_rate', default="0.001", type=float, help="Learning rate")
parser.add_argument('--drop_out', default = 0.3, type=float, help="Drop out")
parser.add_argument('--hidden_units', default=[512, 256], type=int, help="enter 2 integers between 2208 and 102 in decreasing order")
parser.add_argument('--epochs', default=5, type=int, help="set epochs")
parser.add_argument('--gpu', default=False, help="Select GPU")

args = parser.parse_args()

data_dir = args.data_directory
model_name = args.arch
lr = args.learning_rate
save_dir = args.save_dir
hidden_layer1 = args.hidden_units[0]
hidden_layer2 = args.hidden_units[1]
drop_out = args.drop_out
epochs = args.epochs
gpu = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() and gpu==True else "cpu")

if __name__ == "__main__":
    #load data
    trainloader, validationloader, testloader = load_data(data_dir)
    
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
        
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        model.to(device)
     
        epochs = epochs
        steps = 0
        running_loss = 0
        print_every = 10
        print("\nTraining....")   
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(testloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()
    print("--------------------Training finished--------------------")                
    save_checkpoint(save_dir)
    print(f"Save model to: {save_dir}")    
    print("--------------------Saved finished--------------------")           
            