import torch.utils.data as data
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

import numpy as np

import transformers

import tqdm
import gc
import os

import torchvision
import torchvision.transforms as transforms
import PIL

import sys

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_verbose = True

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

bert_version = "bert-large-uncased"
model_name_or_path = 'google/vit-base-patch16-224'

tokenizer = transformers.BertTokenizer.from_pretrained(bert_version)
feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(model_name_or_path)

class BookCoverDataset(data.Dataset):
    def __init__(self, csv_x, csv_y, img_dir, feature_extractor):
        
        # Read the csv file for x
        data_x = pd.read_csv(csv_x)
        
        # Get x: Only reading text part for RNN
        self.x = data_x.iloc[:, 2].values

        # ID of the examples
        self.id = data_x.iloc[:, 0].values
        
        # Get y
        if csv_y is not None:
            data_y = pd.read_csv(csv_y)
            self.y = data_y.iloc[:, 1].values
            self.y = nn.functional.one_hot(torch.tensor(self.y), num_classes=30).reshape(-1, 30)
        else:
            self.y = torch.zeros(len(self.x), 30)

        # Images
        self.images = data_x.iloc[:, 1].values

        # Number of samples
        self.data_len = len(self.x)

        # Image directory
        self.img_dir = img_dir

        # Image feature extractor
        self.feature_extractor = feature_extractor
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        img = PIL.Image.open(img_path)
        image = self.feature_extractor(img, return_tensors="pt")["pixel_values"]
        img.close()
        return self.x[index].lower(), self.y[index], image

    def __len__(self):
        return self.data_len


class CoolModel(nn.Module):
    def __init__(self, text_hidden_size, vision_hidden_size, num_classes):
        super(CoolModel, self).__init__()
        
        self.text = transformers.BertModel.from_pretrained(bert_version)

        self.vision = transformers.ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=-1
        )

        # self.vision = torchvision.models.efficientnet_b3(pretrained=True)

        self.conn = nn.Linear(text_hidden_size + vision_hidden_size, num_classes)

    def forward(self, input_ids, mask, image):
        # Text forward pass
        text_out = self.text(input_ids, attention_mask=mask)["pooler_output"]
        # Vision forward pass
        vision_out = self.vision(image)["logits"]
        
        # Concatenate
        concat_out = torch.cat((text_out, vision_out), dim=1)
        out = self.conn(concat_out)

        return out

def get_predictions(model: CoolModel, dataloader):
    with torch.no_grad():
        actual = torch.Tensor([]).to(device)
        predicted = torch.Tensor([]).to(device)
        for x, y, image in tqdm.auto.tqdm(dataloader):
            input_ids = x["input_ids"]
            mask = x["attention_mask"]
            
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            image = image.to(device)

            output = model(input_ids, mask, image)
            
            y = y.to(device)

            # Get predictions
            predicted = torch.cat((predicted, output), dim=0)
            actual = torch.cat((actual, y), dim=0)
    
        return actual.cpu(), predicted.cpu()

def get_stats(actual, predicted, only_accuracy=True):
    # Accuracy
    accuracy = torch.sum(actual == predicted).item() / len(actual)

    if only_accuracy:
        return {
            'accuracy': accuracy
        }

    # Confusion matrix
    confusion_matrix = torch.zeros(30, 30)
    for i in range(len(actual)):
        confusion_matrix[actual[i]][predicted[i]] += 1
    
    # Precision, recall, f1
    precision = torch.zeros(30)
    recall = torch.zeros(30)
    f1 = torch.zeros(30)
    for i in range(30):
        precision[i] = confusion_matrix[i][i] / torch.sum(confusion_matrix[:, i])
        recall[i] = confusion_matrix[i][i] / torch.sum(confusion_matrix[i, :])
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    # Return dictionary
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(model: CoolModel, dataloader, num_epochs, optimizer, accum_steps=1):
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    cur = 0
    alpha = 0.7
    total = len(dataloader.dataset)
    total_loss = 0
    
    for epoch in range(num_epochs):        
        i = 0
        for x, y, image in tqdm.auto.tqdm(dataloader):
            i += 1
            
            input_ids = x["input_ids"].squeeze(1)
            mask = x["attention_mask"].squeeze(1)

            input_ids = input_ids.to(device)
            mask = mask.to(device)
            image = image.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(input_ids, mask, image)
            loss = criterion(outputs, y.float()) / accum_steps

            total_loss += loss.item()
            
            # Backward and optimize
            loss.backward()
            if i % accum_steps == 0 or i + 1 == total:
                # Print stats
                if global_verbose:
                    print(
                        'Epoch [{}/{}], Loss: {:.4f}'.format(
                            epoch+1, num_epochs, total_loss
                        ),
                        end='\r'
                    )

                optimizer.step()
                optimizer.zero_grad()
                total_loss = 0

def submit(rnn, dataloader: data.DataLoader):
    with torch.no_grad():

        predictions = torch.Tensor([]).to(device)

        for x, y, image in tqdm.auto.tqdm(dataloader):
            input_ids = x["input_ids"].squeeze(1)
            mask = x["attention_mask"].squeeze(1)

            input_ids = input_ids.to(device)
            mask = mask.to(device)
            image = image.to(device)

            output = rnn(input_ids, mask, image)
            
            predictions = torch.cat((predictions, output), dim=0)

        # Save predictions
        predictions = predictions.cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
            
        # Make dataframe
        df = pd.DataFrame({'Id': dataloader.dataset.id, 'Genre': predictions})
        df.to_csv('comp_test_y.csv', index=False)

def collate_fn(batch):    
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    image = [item[2] for item in batch]
    
    # Tokenize
    res = tokenizer(x, return_tensors="pt", padding=True)

    x = {
        'input_ids': res['input_ids'],
        'attention_mask': res['attention_mask']
    }
    
    y = torch.stack(y)
    image = torch.vstack(image)
    
    return x, y, image

dir_path = sys.argv[1]

# Loading the dataset
dataset = BookCoverDataset(f'{dir_path}/train_x.csv', f'{dir_path}/train_y.csv', f"{dir_path}/images/images", feature_extractor)
dataset_test = BookCoverDataset(f'{dir_path}/non_comp_test_x.csv', f'{dir_path}/non_comp_test_y.csv', f"{dir_path}/images/images", feature_extractor)
dataset_sub = BookCoverDataset(f'{dir_path}/comp_test_x.csv', None, f"{dir_path}/images/images", feature_extractor)

batch_size = 12
GRAD_ACCUM_STEPS = 16

dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dataloader_test = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
dataloader_sub = data.DataLoader(dataset_sub, batch_size=256, shuffle=False, collate_fn=collate_fn)

dataloader_all = data.DataLoader(data.ConcatDataset([dataset, dataset_test]), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Create model
text_hidden_size = 1024
vision_hidden_size = 768
num_classes = 30

# Training the model
model = CoolModel(text_hidden_size, vision_hidden_size, num_classes)
model = nn.DataParallel(model.to(device))

# to remove garbage from GPU and empty cache
torch.cuda.ipc_collect()
gc.collect()
torch.cuda.empty_cache()

# The lr needs to be changed after every epoch
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

train_model(model, dataloader, 4, optimizer, GRAD_ACCUM_STEPS)

submit(model, dataloader_sub)