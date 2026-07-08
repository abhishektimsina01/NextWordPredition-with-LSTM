#!/usr/bin/env python
# coding: utf-8

# ## LSTM (Long Short Term Memory):- 
# Long Short term memory is a type of the sequential model, that takes one timestep at a time in order process it and produced two state, cell state and the hidden state.
# 
# Cell staate c(t) is the long term memory, that contains the data which are important for long term. It contains the understanding that the model has understood so far that is important in future timestep and they might depend upon but we dont know which and what will be important.
# 
# Hidden state h(t) is the short term memory, that contains the contextual understanding/summary or the snapshot of the lstm model throughout the process till the previous timestep. It does hold the information that is important and crucial for now.
# 
# It makes the LSTM, different from the RNN because, RNN stores everything in one state due to which the information that we will be loosing is the long term information, which removes the long term dependency/quality information and it causes the lack of prediciton quality.
# 
# Whereas, in LSTM there are two state, cell state only holds the information that are found in early timestep as well as the information that can be crucial for future timesteps. And the short term memory s like the hidden state of rnn, as it holds the contextual understanding till the previous timestep. And the short term memory contains the information crucial in current scenario. And this provides us the ability to play with the LTM and STM like, we can decide what should we forget or is no longer important, what should we consider the information that can be important in future and what are the information we neet to add to STM from LTM which can be crucial for next time step.
# 
# Provides us control to what to forget, what to remember and what to consider right now.

# In[107]:


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter


# In[108]:


nltk.download("punkt_tab")
nltk.download("punkt")


# In[109]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# ### we cannot load all the text from the text.txt file all at once because of the RAM gettig full, so we will use generator to produce desired number of text at a time

# In[110]:


def file():
    with open("smaller.txt", "r") as f:
            for line in f:
                yield line


# In[111]:


def tokens_generator(f):
    tokens = []
    try:
        while True:
            text = next(f)
            tokens.extend(word_tokenize(text.lower()))
    except StopIteration:
        return tokens


# In[112]:


def voccabulary():
    f = file()
    tokens = tokens_generator(f)
    voccab = {
        '<UNK>' : 0,
        '<PAD>' : 1,
    }
    for token in Counter(tokens):
        voccab[token] = len(voccab)
    return voccab


# In[113]:


voccab = voccabulary()


# In[114]:


len(voccab)


# In[115]:


def text_to_indices(voccab):
    f = file()
    dataset = []
    max_len = 0
    # extract text from file
    try:
        while True:
            # tokenize
            indices = []
            text = next(f)
            tokens = word_tokenize(text.lower())
            if max_len < len(tokens):
                max_len = len(tokens)
            # replace with respective integer from voccab
            for token in tokens:
                indices.append(voccab[token])
            # preparing the dataset from indices
            for element in range(len(indices) - 1):
                dataset.append(indices[ : element + 2])
    except StopIteration:
        return  max_len, dataset


# In[116]:


indices = text_to_indices(voccab)


# ### we need to create the dataset (inputs) for the LSTM model
# As we are trying to make the next word predction so, in a sequence there can be different size of sentence so we need to perfrom the padding to make them of the same size as well.

# In[117]:


def dataset_preparatioin(max_length, indices):
    dataset = []
    for i in indices:
        dataset.append([1]*(max_length - len(i)) + i)
    return dataset


# In[118]:


dataset = dataset_preparatioin(indices[0], indices[1])


# ### Now, lets create the dataset and data loader for input and expected output from the model

# In[119]:


class CustomDataSet(Dataset):
    def __init__(self, texts):
        self.features = torch.tensor(texts)[:, :-1].to(device)
        self.target = torch.tensor(texts)[:, -1].to(device)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return (self.features[index], self.target[index])


# In[120]:


data = CustomDataSet(dataset)


# In[121]:


data.__len__()


# In[122]:


data.__getitem__(0)


# In[123]:


dataloader = DataLoader(data, batch_size = 64, shuffle = True)


# In[124]:


for i,j in dataloader:
    print(i,j)
    break


# In[125]:


class NextWordPredictor(nn.Module):

    def __init__(self, voccab):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings = len(voccab), embedding_dim = 64)
        self.lstm = nn.LSTM(input_size = 64, hidden_size = 164, num_layers = 3, batch_first = True)
        self.linear = nn.Linear(164, len(voccab))
    
    def forward(self, features):
        embed = self.embedding(features)
        lstm = self.lstm(embed)[1][0][-1]
        linear = self.linear(lstm)
        return linear


# #### Why do we use the multiple layer in LSTM?
# - As in the case of ANN, during the image classification we need multiple hidden layers because, layer by layer the basic/primitive feature is turned to complex pattern that can classifiy the image.
# - from edges --> structure --> objects --> complex objects --> shape --> item.
# - it starts to see the small part and most basic part of image that is edge and then uses the edges to discover and learn more.
# 
# #### Similarly, in the case of the LSTM, 
# - when we add multiple number of layers in LSTM then the output of one LSTM goes to the input of the other one and so on.
# - Later on training, it slowly first understand the meaning of the word, grammar, pharases and then the whole sentence.
# - At the end of the training, language understanding is developed in the model and ability to really understand the meaning and goal of the text.
# - Its the same as ANN where, it learns the primitive features and the combines them in each layer to make them the complex one able to classify or provide the numberical value.
# - In LSTM, multiple hidden layers allow model to learn the data from base that it its syllable, grammar, phrases. How does different word influences the sentence. It then starts to learn the sentiment and meaninig of the whole sentence when processing step by step.
# 
# internal hidden state of each of the timestep of one layer is input to the following layer stacked and so on. As the hidden state (understanding and summary of all the timestep processed by the model till that timestep) is given, and next layer uses that state for further processing, the primitive understading level ups layer by layer.
# 
# At the end, we get the hidden state of each of the layer at last timestep and we just use the last layer hidden state. As it will contain the overall summary or understadning and whats crucial right now. Unlike the cell state that is an store where all the important topics, understading are stored if may be required in future.
# 
# The hidden state is the LSTM's current contextual understanding of everything it has seen so far. At each timestep, this understanding is updated using the current input and the updated memory in the same timestep(cell state)
# 
# #### Stacking the models is also done in LLM, where we stack the transformers above one another.

# In[126]:


model = NextWordPredictor(voccab)
model.to(device=device)


# In[127]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
epochs = 35


# In[ ]:


for epoch in range(epochs):
    net_loss = 0
    for features, target in dataloader:
        output = model(features)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net_loss = net_loss + loss.item()
    average_loss = net_loss/len(dataloader)
    print("epoch ===>", epoch, "loss ===>", average_loss)

    

