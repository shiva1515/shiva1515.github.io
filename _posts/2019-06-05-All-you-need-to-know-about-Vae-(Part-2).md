---
layout: post
title: "All you need to know about Variational AutoEncoder(Part-2)"
date: 2019-06-05
mathjax: True
---
In this part, we will see how using **Variational AutoEncoder** we generate a similar type of text; We use VAE in a molecular generation.
If you don’t know about VAE, please visit the part-1 of <a href="https://shiva1515.github.io/2019/06/04/All-you-need-to-know-about-Vae-(Part-1).html" target="_blank">VAE</a> you will understand the theory part better.<br/>
Molecular generation is the generating of new molecules. In this model, we generate a similar molecule as input molecules.input of this model is smiles, and the output is a similar type of smiles as of input. We use pytorch to implement VAE in a molecular generation.<br/>
<br/>


{% include image.html url="/assets/img/main_image.jpg" description="" %}
<br/>
<br/>
**SMILES** (Simplified Molecular Input Line Entry System) are the type of chemical notation that helps us to represent molecules and easy to used by the computers. It is a simple string representation of molecules. Some examples
c1ccccc1 benzene,
c1c(N(=O)=O)cccc1 Nitrobenzene

{% include image.html url="/assets/img/vae-2.png" description="" %}

Here we see molecular generation in pytorch.For the implementation of VAE in the molecular generation, we will we using ChEMBL smiles dataset which contains 2M smiles, and it is a manually curated database of bioactive drug-like molecules.<br/>
We will be using RNN in this model because RNN tries to capture the pattern of text easily as compared to CNN and also in RNN, we have LSTM and GRU for memorizing.<br/>
We use the GRU (Gated Recurrent Unit) model because it aims to solve the vanishing gradient problem, which comes with a standard recurrent neural network.<br/>

*Let's get hands dirty on the code.*<br/>
**Import dataset**<br/>
the dataset we used for this purpose is refined from Zinc Database <a href="https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv" target="_blank">dataset</a><br/>
This is huge dataset so first slice the dataset according to your system.this dataset contain 1,936k smiles.
we use only 1k smiles for faster computation.<br/>
First, we will import our dataset, which contains smiles and convert into the strings.

``` python
import pandas as pd
import torch


data = pd.read_csv('.../dataset_iso_v1.csv')
train_data1 = data[data['SPLIT'] == 'train']
train_data_smiles2 = (train_data1["SMILES"].squeeze()).astype(str).tolist()
train_data = train_data_smiles2
```

**Building a vocabulary**<br/>
After that, we build a vocabulary for the model, to make vocabulary we use set() because when we feed data into the set, it removes the repeat data. After that, we make a list of characters to make vocabulary for our model than we add <’bos’> to indicate the beginning of smiles,<’eos’> to indicate the end of a sentence,<”pad”> to make all smiles of the same length and <”unk”> for unknown words.<br/>
<br/>
After completion of vocabulary we make character to index and index to character, to encode decode the vocabulary.
To decrease the time for training the model, we use Cuda in pytorch; this is for changing our computation from CPU to GPU because GPU handles lots of parallel computations using thousands of cores. Also, they have a large memory bandwidth to deal with the data for these computations.<br/>
after completion of vocabulary we then feed into the embedding matrix; we will discuss about embedding matrix later.
We create several functions which we use later in the model character to ids, ids to character, string to ids, ids to string, string to tensor. You can see the functions, and you understand why we create these functions.<br/>

```python
chars = set()
for string in train_data:
    chars.update(string)
all_sys = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
vocab = all_sys
c2i = {c: i for i, c in enumerate(all_sys)}
i2c = {i: c for i, c in enumerate(all_sys)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vector = torch.eye(len(c2i))


def char2id(char):
    if char not in c2i:
        return c2i['<unk>']
    else:
        return c2i[char]


def id2char(id):
    if id not in i2c:
        return i2c[32]
    else:
        return i2c[id]

def string2ids(string,add_bos=False, add_eos=False):
    ids = [char2id(c) for c in string]
    if add_bos:
        ids = [c2i['<bos>']] + ids
    if add_eos:
        ids = ids + [c2i['<eos>']]
    return ids
def ids2string(ids, rem_bos=True, rem_eos=True):
    if len(ids) == 0:
        return ''
    if rem_bos and ids[0] == c2i['<bos>']:
        ids = ids[1:]
    if rem_eos and ids[-1] == c2i['<eos>']:
        ids = ids[:-1]
    string = ''.join([id2char(id) for id in ids])
    return string
def string2tensor(string, device='model'):
    ids = string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(ids, dtype=torch.long,device=device if device == 'model' else device)
    return tensor
tensor = [string2tensor(string, device=device) for string in train_data]
vector = torch.eye(len(c2i))
```

**Model**<br/>
In the previous post about VAE part, I explain the role of the encoder, decoder and latent vector.As we now create vae model, if you have doubts in the theory part of VAE, you can see my previous post.<br/>

We feed our vocabulary and one hot matrix into the embedding matrix.
Embedding is a way to represent a character or word into a dense vector representation.embeddings are useful because they can reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space.

We then feed our embedding dimension into the encoder. The encoder encodes our input sample data into small vector from which we call latent vector. This small vector represents our entire dataset.
The encoder output gives latent space vector and kl divergence loss
The decoder work is to decode the latent space and bring back our input data.it is not necessary that our encoder and decoder is the same we can change the layers according to us.
The latent sample takes probability distributions of various characteristics of our input. Let say we have a latent vector of 3 nodes this means our latent space vector define our input into these 3 nodes and these 3 nodes contain a distribution of characteristics of the input dataset.
We take a sample data from this distribution
the whole code of VAE model is:
``` python
import torch.nn as nn
import torch.nn.functional as F

q_bidir = True
q_d_h = 256
q_n_layers = 1
q_dropout = 0.5
d_n_layers = 3
d_dropout = 0
d_z = 128
d_d_h = 512
from data import *
class VAE(nn.Module):
  def __init__(self,vocab,vector):
    super().__init__()
    self.vocabulary = vocab
    self.vector = vector
    
    n_vocab, d_emb = len(vocab), vector.size(1)
    self.x_emb = nn.Embedding(n_vocab, d_emb, c2i['<pad>'])
    self.x_emb.weight.data.copy_(vector)
  
    #ENCODER
    
    self.encoder_rnn = nn.GRU(d_emb,q_d_h,num_layers=q_n_layers,batch_first=True,dropout=q_dropout if q_n_layers > 1 else 0,bidirectional=q_bidir)
    q_d_last = q_d_h * (2 if q_bidir else 1)
    self.q_mu = nn.Linear(q_d_last, d_z)
    self.q_logvar = nn.Linear(q_d_last, d_z)
  
  
  
    # Decoder
    self.decoder_rnn = nn.GRU(d_emb + d_z,d_d_h,num_layers=d_n_layers,batch_first=True,dropout=d_dropout if d_n_layers > 1 else 0)
    self.decoder_latent = nn.Linear(d_z, d_d_h)
    self.decoder_fullyc = nn.Linear(d_d_h, n_vocab)
  
  
  
    # Grouping the model's parameters
    self.encoder = nn.ModuleList([self.encoder_rnn,self.q_mu,self.q_logvar])
    self.decoder = nn.ModuleList([self.decoder_rnn,self.decoder_latent,self.decoder_fullyc])
    self.vae = nn.ModuleList([self.x_emb,self.encoder,self.decoder])
    
    
    
  @property
  def device(self):
    return next(self.parameters()).device

  def string2tensor(self, string, device='model'):
    ids = string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(ids, dtype=torch.long,device=self.device if device == 'model' else device)
    return tensor

  def tensor2string(self, tensor):
    ids = tensor.tolist()
    string = ids2string(ids, rem_bos=True, rem_eos=True)
    return string
  
  def forward(self,x):
    z, kl_loss = self.forward_encoder(x)
    recon_loss = self.forward_decoder(x, z)
    print("forward")
    return kl_loss, recon_loss
  
  def forward_encoder(self,x):
    x = [self.x_emb(i_x) for i_x in x]
    x = nn.utils.rnn.pack_sequence(x)
    _, h = self.encoder_rnn(x, None)
    h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
    h = torch.cat(h.split(1), dim=-1).squeeze(0)
    mu, logvar = self.q_mu(h), self.q_logvar(h)
    eps = torch.randn_like(mu)
    z = mu + (logvar / 2).exp() * eps
    kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
    return z, kl_loss
  
  def forward_decoder(self,x, z):
    lengths = [len(i_x) for i_x in x]
    x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value= c2i['<pad>'])
    x_emb = self.x_emb(x)
    z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
    x_input = torch.cat([x_emb, z_0], dim=-1)
    x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
    h_0 = self.decoder_latent(z)
    h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
    output, _ = self.decoder_rnn(x_input, h_0)
    output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
    y = self.decoder_fullyc(output)
    
    recon_loss = F.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)),x[:, 1:].contiguous().view(-1),ignore_index= c2i['<pad>'])

  
    
  def sample_z_prior(self,n_batch):
    return torch.randn(n_batch,self.q_mu.out_features,device= self.x_emb.weight.device)
  def sample(self,n_batch, max_len=100, z=None, temp=1.0):
    with torch.no_grad():
      if z is None:
        z = self.sample_z_prior(n_batch)
        z = z.to(self.device)
        z_0 = z.unsqueeze(1)
        h = self.decoder_latent(z)
        h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        w = torch.tensor(c2i['<bos>'], device=self.device).repeat(n_batch)
        x = torch.tensor([c2i['<pad>']], device=device).repeat(n_batch, max_len)
        x[:, 0] = c2i['<bos>']
        end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
        eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device)


        for i in range(1, max_len):
          x_emb = self.x_emb(w).unsqueeze(1)
          x_input = torch.cat([x_emb, z_0], dim=-1)

          o, h = self.decoder_rnn(x_input, h)
          y = self.decoder_fullyc(o.squeeze(1))
          y = F.softmax(y / temp, dim=-1)

          w = torch.multinomial(y, 1)[:, 0]
          x[~eos_mask, i] = w[~eos_mask]
          i_eos_mask = ~eos_mask & (w == c2i['<eos>'])
          end_pads[i_eos_mask] = i + 1
          eos_mask = eos_mask | i_eos_mask
          
          
          new_x = []
          for i in range(x.size(0)):
            new_x.append(x[i, :end_pads[i]])

            
    return [self.tensor2string(i_x) for i_x in new_x]
    
```

**Training**<br/>
We use inbuilt pytorch function DataLoader to create batches and keep similar length smiles into the same batch. In dataloader we have collate function which helps to convert data into tensors and do padding to make similar length data.it is not necessary to use this collate function if we have a same length of data or we convert our data into tensors already.<br/>
Then the data coming from dataloader we use it to the training process.In this training, We use Adam optimizer for optimization, and then we train our data into 50 epoch(one epoch means the whole model run 1 time)<br/>
By using this, we try to minimize KL-divergence loss and Reconstruction loss. I explain both losses in my previous post,
Reconstruction loss is how well our model reconstructs back the input data and kl-divergence loss measure how much information is lost when using q to represent p. Remember these q and p is data distribution.<br/>

```python
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import math
import numpy as np
from collections import UserList, defaultdict
n_last = 1000
n_batch = 32
kl_start = 0
kl_w_start = 0.0
kl_w_end = 1.0
n_epoch = 50
n_workers = 0

clip_grad  = 50
lr_start = 0.003
lr_n_period = 10
lr_n_mult = 1
lr_end = 3 * 1e-4
lr_n_restarts = 6
from data import *

def _n_epoch():
    return sum(lr_n_period * (lr_n_mult ** i) for i in range(lr_n_restarts))
  
def _train_epoch(model, epoch, train_loader, kl_weight, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
      
    kl_loss_values = CircularBuffer(n_last)
    recon_loss_values = CircularBuffer(n_last)
    loss_values = CircularBuffer(n_last)
    for i, input_batch in enumerate(train_loader):
        input_batch = tuple(data.to(device) for data in input_batch)
      
    #forward
        kl_loss, recon_loss = model(input_batch)
        loss = kl_weight * kl_loss + recon_loss
    #backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(get_optim_params(model),clip_grad)
            optimizer.step()
      
        kl_loss_values.add(kl_loss.item())
        recon_loss_values.add(recon_loss.item())
        loss_values.add(loss.item())
        lr = (optimizer.param_groups[0]['lr'] if optimizer is not None else None)
      
    #update train_loader
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        postfix = [f'loss={loss_value:.5f}',f'(kl={kl_loss_value:.5f}',f'recon={recon_loss_value:.5f})',f'klw={kl_weight:.5f} lr={lr:.5f}']
    postfix = {'epoch': epoch,'kl_weight': kl_weight,'lr': lr,'kl_loss': kl_loss_value,'recon_loss': recon_loss_value,'loss': loss_value,'mode': 'Eval' if optimizer is None else 'Train'}
    return postfix
  
def _train(model, train_loader, val_loader=None, logger=None):
    optimizer = optim.Adam(get_optim_params(model),lr= lr_start)
    
    lr_annealer = CosineAnnealingLRWithRestart(optimizer)
    
    model.zero_grad()
    for epoch in range(n_epoch):
      
        kl_annealer = KLAnnealer(n_epoch)
        kl_weight = kl_annealer(epoch)
        postfix = _train_epoch(model, epoch,train_loader, kl_weight, optimizer)
        lr_annealer.step()
def fit(model, train_data, val_data=None):
    logger = Logger() if False is not None else None
    train_loader = get_dataloader(model,train_data,shuffle=True)

    
    
    val_loader = None if val_data is None else get_dataloader(model, val_data, shuffle=False)
    _train(model, train_loader, val_loader, logger)
    return model
def get_collate_device(model):
    return model.device
def get_dataloader(model, train_data, collate_fn=None, shuffle=True):
    if collate_fn is None:
        collate_fn = get_collate_fn(model)
        print(collate_fn)
    return DataLoader(train_data, batch_size=n_batch, shuffle=shuffle, num_workers=n_workers, collate_fn=collate_fn)

def get_collate_fn(model):
    device = get_collate_device(model)

    def collate(train_data):
        train_data.sort(key=len, reverse=True)
        tensors = [string2tensor(string, device=device) for string in train_data]
        return tensors

    return collate

def get_optim_params(model):
    return (p for p in model.parameters() if p.requires_grad)

class KLAnnealer:
    def __init__(self,n_epoch):
        self.i_start = kl_start
        self.w_start = kl_w_start
        self.w_max = kl_w_end
        self.n_epoch = n_epoch

        
        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc
      
      
      
class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self , optimizer):
        self.n_period = lr_n_period
        self.n_mult = lr_n_mult
        self.lr_end = lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end
            
            
            
            
class CircularBuffer:
    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]

    def mean(self):
        return self.data.mean()
      
      
class Logger(UserList):
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return Logger(self.data[key])
        else:
            ldata = self.sdata[key]
            if isinstance(ldata[0], dict):
                return Logger(ldata)
            else:
                return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)
           
```

**Sample from model**<br/>
for taking a sample from the model we use the encoder and take mean and sigma from the encoder after taking mean and sigma we convert this into the decoder dimension, and we get sample smiles.<br/>
We first feed <”bos”> to the decoder then this decoder tries to give one character at a time we run this in a for loop from 1 to maximum length from this we obtain a list of tensor each tensor represent the character of smile and then by  using tensortostring function we convert obtained tensor to smiles string.<br/>
We give the maximum length up to which we generate smiles if it generates less then maximum we pad it, and after converting back to string from tensor, we remove this padding.<br/>

```python
from tqdm import tqdm
import pandas as pd
n_samples = 3000
n_jobs = 1
max_len = 100

class sample():
  def take_samples(model,n_batch):
    n = n_samples
    samples = []
    with tqdm(total=n_samples, desc='Generating samples') as T:
      while n > 0:
        current_samples = model.sample(min(n, n_batch), max_len)
        samples.extend(current_samples)
        n -= len(current_samples)
        T.update(len(current_samples))
    samples = pd.DataFrame(samples, columns=['SMILES'])
    return samples
```
I used pycharm for writing this code. all above code of model,trainer,data preprocessing,samples are of seperate files so finally to run the code we have a file name run.py and the code for running the model is given below.<br/>
```python
from trainer import *
from vae_model import VAE
from data import *
from samples import *

model = VAE(vocab,vector).to(device)
fit(model, train_data)
model.eval()
sample = sample.take_samples(model,n_batch)
print(sample)
```

hope you like it. if you want to know more about molecular generation please visit <a href="https://github.com/bayeslabs" target="_blank">bayes lab</a> github page here you will find more models on molecular generation.<br/>

stay tuned in the next blog we will see about Adversarial Autoencoders.

If you have any doubts please Let us know in the comments.<br/>
Thank you 

References<br/>

I use <a href="https://github.com/molecularsets/moses/tree/master/moses/vae" target="_blank">github</a>  this GitHub code in this blog to explain VAE. you can see more models in molecular generation in pytorch into this GitHub page.










