# Augmented data
The augmented data of each proposed approach is available in .csv format in the below links :

[Source](https://drive.google.com/file/d/1LB0MOzpN8lGovKIEhUgilDpnNjBRHYfZ/view?usp=sharing)

[Token](https://drive.google.com/file/d/10xSxu5bCp34yV0uGi91ZfY55djboICdK/view?usp=sharing)

[Mono](https://drive.google.com/file/d/1JrpO-YLxBvN6PekGTcrfgkZk8_vbqwFS/view?usp=sharing)

[Replace](https://drive.google.com/file/d/1yNw8ImqafnN-ndjKaWsqz0eGAKF4ngav/view?usp=sharing)

[Reverce](https://drive.google.com/file/d/14q2X_gp3hXo5hU0rw3WQbEc720B1-g_Z/view?usp=sharing)

[Swap](https://drive.google.com/file/d/1Hg6nMXaoxBUPfIpg3rzgVBCZIlvnhOsX/view?usp=sharing)

[Misspelling](https://drive.google.com/file/d/1tvHfX4qNFR3OeItAW8RtiG6mHXRf-dZx/view?usp=sharing)

[DAGEC](https://drive.google.com/file/d/15zq8TvnKzVSAWX3k2uatadCC2OdpDUdW/view?usp=sharing)




# Preparing the Data

Import all the required modules and packages.
 
```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import TabularDataset, Field, Iterator, BucketIterator, ReversibleField

import pyarabic.araby as araby
import pyarabic.number as number

import bpemb
from bpemb import BPEmb
bpemb_ar = BPEmb(lang="ar", vs=1000, dim=300) 
```
Create the tokenizer using bpemb as belllow:

```py
def normalizeString(line):
    line = araby.strip_tatweel(line)
    line = araby.strip_tashkeel(line)
    line = bpemb_ar.encode(line)
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

train_data, valid_data = TabularDataset.splits(path='../data/',train='AGEC_Training_set.csv',
    validation='AGEC_development_set.csv' , format='csv',
    fields=[('src', SRC), ('trg', TRG)], skip_header=True) 
