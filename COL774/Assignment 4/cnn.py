import torch 
import os
import numpy as np
import pandas as pd
import random
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, NLLLoss, LogSoftmax, CrossEntropyLoss
from torch import flatten
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision import datasets,  transforms, models


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Constructing the CNN architecture
class CNN_arch(Module):
    def __init__(self, num_channels, ):
        super(CNN_arch, self).__init__()
        self.conv1 = Conv2d(in_channels=num_channels, out_channels = 32, kernel_size=(5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2))

        self.conv2 = Conv2d(in_channels = 32, out_channels = 64, kernel_size=(5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2))

        self.fc1 = Linear(in_features = 73728, out_features=128)
        self.relu4 = ReLU()
        
        self.fc2 = Linear(in_features = 128, out_features=30)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output

# Defining a CustomDataset Loader
class CustomDataset(Dataset):
    def __init__(self, desc_file, label_file, img_dir, transform = None):
        self.img_labels = pd.read_csv(label_file)
        self.img_desc = pd.read_csv(desc_file)
        self.img_dir = img_dir 
        self.transform = transform 
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_desc.iloc[idx, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# setting the required_grad to False
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# predicting on dataloader
def predict(dataLoader, model, device, out="submission.csv"):
    num_examples = len(dataLoader.dataset)
    correct = 0
    with torch.no_grad():
        predictions = torch.Tensor([]).to(device)
        # set the model in evaluation mode
        model.eval()

        # loop over the test set
        for (images, labels) in dataLoader:
            # send the input to the device
            images = images.to(device)
            labels = labels.to(device)
            # make the predictions and add them to the list
            outputs = model(images)
            # correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            predictions = torch.cat((predictions, output), dim=0)

        # Save predictions
        predictions = predictions.cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
            
        # Make dataframe
        df = pd.DataFrame({'Id': dataloader.dataset.id, 'Genre': predictions})
        df.to_csv(out, index=False)

# training function
def train_model(dataloader, model, device, epochs, lr):
    
    model= nn.DataParallel(model)
    model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for e in range(0, epochs):

        model.train()
        train_correct = 0
        test_correct = 0
        for (images, labels) in dataloader['train']:
            (images, labels) = (images.to(device), labels.to(device))
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            
        # test_correct = predict(dataloader['test'], model, device)
        # print("Epoch {}, Training accuracy: {}/{}, Loss: {}".format(e, train_correct, len(dataloader['train'].dataset) , loss.item()))
        # print("Epoch {}, Test accuracy: {}/{}".format(e, test_correct, len(dataloader['test'].dataset)))

import sys
dir_path = sys.argv[1]

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
			transforms.Normalize(mean=[0.5482, 0.5109, 0.4749],std=[0.3384, 0.3237, 0.3218])])

train_data = CustomDataset(f'{dir_path}/train_x.csv', 
							f'{dir_path}/train_y.csv', 
							f'{dir_path}/images/images/', transform)

test_data = CustomDataset(f'{dir_path}/non_comp_test_x.csv', 
						f'{dir_path}/non_comp_test_y.csv', 
						f'{dir_path}/images/images/', transform)

model = CNN_arch(num_channels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 500
lr = 0.001
dataloader = {}
dataloader['train'] = DataLoader(train_data, shuffle = True, batch_size=batch_size)
dataloader['test'] = DataLoader(test_data, batch_size=batch_size)

train_model(dataloader, model, device, epochs=25, lr=lr)
predict(dataloader['test'], model, device, 'non_comp_test_pred_y.csv')
