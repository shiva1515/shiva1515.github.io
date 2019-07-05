---
layout: post
title: "All you need to know about Variational AutoEncoder(Part-2)"
date: 2019-06-05
mathjax: True
---
In this part, we will see how using **Variational AutoEncoder** we generate a similar type of text; We use VAE in a molecular generation.
If you don’t know about VAE, please visit the part-1 of VAE you will understand the theory part better.<br/>
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
First, we will import our dataset, which contains smiles and convert into the strings.

``` python
import pandas as pd
import torch


data = pd.read_csv('C:/Users/ASUS/Desktop/intern things/dataset_iso_v1.csv')
train_data1 = data[data['SPLIT'] == 'train']
train_data_smiles2 = (train_data1["SMILES"].squeeze()).astype(str).tolist()
train_data = train_data_smiles2
```

**Building a vocabulary**
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

**Model**
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







