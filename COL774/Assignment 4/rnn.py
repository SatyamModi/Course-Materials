# %%
import torch.utils.data as data
import pandas as pd
import torchtext
import torch
import torch.nn as nn
import numpy as np
from collections import Counter, OrderedDict

from nltk.corpus import stopwords

# %%
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
global_verbose = True

# %%
class BookCoverDataset(data.Dataset):
    def __init__(self, csv_x, csv_y=None):
        
        # Read the csv file
        data_x = pd.read_csv(csv_x)
        # Get x: Only reading text part for RNN
        self.x = data_x.iloc[:, 2].values
        
        # ID of the examples
        self.id = data_x.iloc[:, 0].values
        
        # Read the csv file
        if csv_y is not None:
            data_y = pd.read_csv(csv_y)
            # Get y
            self.y = data_y.iloc[:, 1].values
            self.y = nn.functional.one_hot(torch.tensor(self.y), num_classes=30).reshape(-1, 30)

        # Number of samples
        self.data_len = len(self.x)

        # Pre process the data
        self.x = self.pre_process_x(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.data_len

    def pre_process_x(self, x):
        # Tokenize the text
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        x = [tokenizer(i) for i in x]

        # Remove punctuation from x
        x = [[i for i in j if str.isalnum(i)] for j in x]

        # Remove stopwords from x
        stop_words = set(stopwords.words('english'))
        x = [[i for i in j if i not in stop_words] for j in x]
        return x

# %%
def build_vocab(x):    
    # Build the vocabulary
    counter = Counter(
        [token for tokens in x for token in tokens]
    )

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab = torchtext.vocab.vocab(ordered_dict, specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    return vocab

# %%
def transform_from_vocab(x, vocab):
    for i, tokens in enumerate(x):
        x[i] = [vocab[token] for token in tokens]
    return x

# %%
def collate_fn(batch):
    # Padding
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(i[0]) for i in batch], batch_first=True), torch.stack([i[1] for i in batch])

# %%
class MyRNN(nn.Module):
    vocab = None
    glove = None

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dataset: BookCoverDataset):
        super(MyRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Build pre-trained embedding from glove
        # glove, vocab: Global Variables
        mat = torch.zeros(len(MyRNN.vocab), 300)
        for word in MyRNN.vocab.get_stoi():
            if word in MyRNN.glove.stoi:
                mat[MyRNN.vocab[word]] = MyRNN.glove[word]
            else:
                mat[MyRNN.vocab[word]] = torch.randn(300)
        
        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(mat, freeze=False)

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional = True, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x.long())

        # Forward propagate RNN
        hT, _ = self.rnn(x)

        # Decode the hidden state of the last time step
        # Use tanh activation
        out = self.fc1(hT[:, 0, :])
        out = torch.tanh(out)

        out = self.fc2(out)
        out = torch.sigmoid(out)

        return out

# %%
def get_accuracy(rnn: MyRNN, dataloader):
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = rnn(x)
            
            predicted = torch.argmax(outputs.data, 1)
            actual = torch.argmax(y.data, 1)
            
            total += actual.size(0)
            correct += (predicted == actual).sum().item()

        return 100 * correct / total

# %%
import tqdm

def train_rnn(rnn: MyRNN, dataloader, num_epochs, learning_rate):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate, weight_decay=1e-5)
    rnn.train()
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        for (x, y) in dataloader:
            # Move tensors to the configured device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            outputs = rnn(x)
            loss = criterion(outputs, y.float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# %%
def submit(rnn, dataloader: data.DataLoader, out="submission.csv"):
    with torch.no_grad():

        predictions = torch.Tensor([]).to(device)

        for x, _ in tqdm.auto.tqdm(dataloader):
            x = x.to(device)
            output = rnn(x)
            
            predictions = torch.cat((predictions, output), dim=0)

        # Save predictions
        predictions = predictions.cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
            
        # Make dataframe
        df = pd.DataFrame({'Id': dataloader.dataset.id, 'Genre': predictions})
        df.to_csv(out, index=False)

# %%
# dataset = BookCoverDataset('data/train_x.csv', 'data/train_y.csv')
# test_dataset = BookCoverDataset('data/non_comp_test_x.csv', 'data/non_comp_test_y.csv')

import sys
dir_path = sys.argv[1]
# dir_path = "../input/col774-2022"

dataset = BookCoverDataset(f'{dir_path}/train_x.csv', f'{dir_path}/train_y.csv')
test_dataset = BookCoverDataset(f'{dir_path}/non_comp_test_x.csv', f'{dir_path}/non_comp_test_y.csv')

# %%
# Creating Vocabulary from training data
MyRNN.vocab = build_vocab(dataset.x)

# Glove embeddings
MyRNN.glove = torchtext.vocab.GloVe(name='6B', dim=300)

# %%
# Transforming x
dataset.x = transform_from_vocab(dataset.x, MyRNN.vocab)
test_dataset.x = transform_from_vocab(test_dataset.x, MyRNN.vocab)

# %%
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=collate_fn)

# %%
input_size = 300
hidden_size = 128
num_layers = 1
num_classes = 30

rnn = MyRNN(input_size, hidden_size, num_layers, num_classes, dataset).to(device)

# %%
train_rnn(rnn, dataloader, 50, 0.0005)

# %%
submit(rnn, test_dataloader, 'non_comp_test_pred_y.csv')

# %%
