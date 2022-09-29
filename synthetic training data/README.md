# Training data
The synthetic data of the proposed approach is available in .csv format in the below links :

[Training set](https://drive.google.com/file/d/1YUYaLAoPBU1HyUy95qY7cWll7-bq_x92/view?usp=sharing) 

[Development set](https://drive.google.com/file/d/1u5D7UBWhgkDVdcTujm7Hc8LXbD9Dhf6N/view?usp=sharing) 

[Test set](https://drive.google.com/file/d/1ZH0R_pzf96wRLgk97v6bWTNy0-uVIv_l/view?usp=sharing)

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
