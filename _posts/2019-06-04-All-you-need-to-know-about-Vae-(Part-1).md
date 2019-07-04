---
layout: post
title: "All you need to know about Vae (Part-1)"
date: 2019-06-04
mathjax: True
---

{% include image.html url="/assets/img/vae_part_1.jpg" description="" %} 
In this part of the blog, we will discuss the basics of the \textbf{Variational AutoEncoder}(VAE). And in the next section, we will implement VAE in a molecular generation.
Vae is a type of generative model which helps us to generate a similar type of input data. It helps to generate similar images, similar text etc.
A generative model is a way of learning similar data distribution of input data so that it generates new similar type of data.
VAEs also make a probability distribution of input data, and from that distribution, we create samples which is taking data from this distribution and generate new data similar to input data.

{% include image.html url="/assets/img/vae-gaussian.png" description="" %}

So as we see from the above diagram, Vae has mainly 3 components or says we divide the VAE into three parts for better understanding of vae.
Encoder: Encoder is a neural network that takes input data, and it converts higher dimensional data into lower dimensional data which we call latent space. Lets say we have an image of 28*28(784) pixels what encoder does is it convert our 784 dimensional images into a small dimensions of lets say 8 so the encoder tries to pass information of whole 784 dimension images to 8 dimension vector it encodes in such a way that this 8 dimensional space represents our whole input data.
In Vae we do not say encoder we say probabilistic encoder because in Vae the small dimensional latent space does not take a discrete range of values it takes a probability distribution. As above we say we have an 8-dimensional small vector then 8 nodes represent some character of input data. e.g. if our input data is human faces, then these nodes may represent smiles, eyes shape, etc. and create a probability distribution of these characters.

We represent encoder as $q_\phi(z|x)$ which means find the z(small dimension latent space) given x which is input data. In general case, we take $q_\phi(z|x)$ is Gaussian distribution you can take any distribution whose distribution you know.we will discuss it later.
{% include image.html url="/assets/img/encoder-decoder.png" description="" %}

**Latent Space:** It is a layer in the neural network which represent our whole input data. It is also called bottleneck because of in very small dimension it represents whole data.

**Decoder:** As you see in below diagram you understand what decoder role in VAE, it converts latent sample space back to our input data. It converts back our 8-dimensional latent space into the 784-dimensional image.
We represent decoder as $p_\theta(x|z)$ which means to find x provided z.

{% include image.html url="/assets/img/vae.jpg" description="" %}
**Goal of Vae**
The goal of VAE is to find gaussian distribution $q_\phi(z|x)$ and take a sample from z ~ $q_\phi(z|x)$ (sampling z from $q_\phi(z|x)$) and generate some similar output.
Why we use Gaussian in VAE encoder
You may notice in encoder section we use Gaussian distribution in the encoder, so first I clear some point why we take a known distribution in encoder region.
Let x be the input and z be the set of latent variables with joint distribution $p(z,x)$  the problem is to compute the conditional distribution of z given x $p(z|x)$

To compute $p(z\mid x)=\frac{p(x\mid z)\, p(z)}{p(x)}$ we have to compute the $p(x)=\int_{z} p(x,z) dx$ but the integral is not available in closed form or is intractable(i.e require exponential time to compute) because of multiple  variable involved in latent vector z.

To remove this problem, we use an alternative solution, which is we approximate $p(z\mid x)$ with some known distribution $q(z\mid x)$ which is tractable. This is done by Variational Inference(VI)
We use KL-divergence to approximate the $p(z\mid x)$ and $q(z\mid x)$.this divergence measures how much information is lost when using q to represent p. It is one measure q close to p. And we try to minimize the KL-divergence so to get similar distribution.

-$$D_{kl}(q_\phi(z\mid x)||p_\theta(z\mid x)) = sum_  q_\theta(z\mid x)log\frac{q_\phi(z\mid z)}{p_\theta(z\mid x)} $$


















