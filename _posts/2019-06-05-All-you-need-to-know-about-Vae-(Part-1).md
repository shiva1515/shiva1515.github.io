---
layout: post
title: "All you need to know about Variational AutoEncoder(Part-2)"
date: 2019-06-05
mathjax: True
---
In this part, we will see how using **Variational AutoEncoder** we generate a similar type of text; We use VAE in a molecular generation.
If you don’t know about VAE, please visit the part-1 of VAE you will understand the theory part better.<br/>
Molecular generation is the generating of new molecules. In this model, we generate a similar molecule as input molecules.input of this model is smiles, and the output is a similar type of smiles as of input. We use pytorch to implement VAE in a molecular generation.<br/>


{% include image.html url="/assets/img/main_image.jpg" description="" %}

**SMILES** (Simplified Molecular Input Line Entry System) are the type of chemical notation that helps us to represent molecules and easy to used by the computers. It is a simple string representation of molecules. Some examples
c1ccccc1 benzene
c1c(N(=O)=O)cccc1 Nitrobenzene

{% include image.html url="/assets/img/vae-2.png" description="" %}

Here we see molecular generation in pytorch.For the implementation of VAE in the molecular generation, we will we using ChEMBL smiles dataset which contains 2M smiles, and it is a manually curated database of bioactive drug-like molecules.<br/>
We will be using RNN in this model because RNN tries to capture the pattern of text easily as compared to CNN and also in RNN, we have LSTM and GRU for memorizing.<br/>
We use the GRU (Gated Recurrent Unit) model because it aims to solve the vanishing gradient problem, which comes with a standard recurrent neural network.<br/>

*Let's get hands dirty on the code.*
**Import dataset**
First, we will import our dataset, which contains smiles and convert into the strings.

``` python
import pandas as pd
import torch


data = pd.read_csv('C:/Users/ASUS\Desktop/intern things/dataset_iso_v1.csv')
train_data1 = data[data['SPLIT'] == 'train']
train_data_smiles2 = (train_data1["SMILES"].squeeze()).astype(str).tolist()
train_data = train_data_smiles2
```



