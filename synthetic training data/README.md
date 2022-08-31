# Training data
The synthetic data of the proposed approach is available in .csv format in the below links :

[Training_set](https://drive.google.com/file/d/1zrjQufjgJsfTzfCoU2U1rmUc0TIhijHh/view)

[Development_set](https://drive.google.com/file/d/14y0TTTs8qXBLHonNKgbgwXYxEGx48D3x/view?usp=sharing)


# Preparing the Data

Import all the required modules and packages.
 
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import TabularDataset, Field, Iterator, BucketIterator, ReversibleField

import bpemb
from bpemb import BPEmb
bpemb_id = BPEmb(lang="id", vs=1000, dim=300) 
```
Create the tokenizer using bpemb as belllow:

```py
def normalizeString(line):
    line = bpemb_id.encode(line)
    return line
```

Then create fields, the model expects data to be fed with in fromat of the batch dimension first, so we use batch_first = True:

```py
SRC = Field(tokenize= normalizeString, init_token='<sos>', eos_token='<eos>',  batch_first=True) 
TRG = Field(tokenize= normalizeString, init_token='<sos>', eos_token='<eos>',  batch_first=True) 
```

Next, build the vocabulary and load the dataset:

````py
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

train_data, valid_data = TabularDataset.splits(path='../data/',train='Training_set.csv',
    validation='Development_set.csv' , format='csv',
    fields=[('src', SRC), ('trg', TRG)], skip_header=True) 
