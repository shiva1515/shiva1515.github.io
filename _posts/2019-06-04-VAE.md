---
layout:blog
title:"lets see"
date:"2019-06-04"
mathjax:True
---


in my past two blogs you see how vae is helpful in creating SMILES(text generation) of similar kind.
now we see there are many model which work on same as VAE does but whats the difference between VAE and others it they both work on same things
today we will see about drawbacks of VAE and how Generative adversiral network and Adversiral autoencoder is better than Variational autoencoder.
the assumption we took in VAE is that we used another gaussian distribution and imposed this distribution to our latent vector distribution because we dont know the 
distribution of input data and to do this we use KL-divergence to make similar distribution.
we take assumption that the two distribution will overlap each other if this not happen then our KL-divergence gives the $\infty$ which gives 
some weired results. lets understand this by an simple example

lets say Q
