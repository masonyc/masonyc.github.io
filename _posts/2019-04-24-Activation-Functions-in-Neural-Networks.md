---
title: "Activation Functions in Neural Networks"
date: 2019-04-24 20:27:20
tags: 
  - Machine Learning
  - Deep Learning
  - Neural Networks
categories: "AI" 
---
In this article, I am going to talk about the commonly used activation functions in neural networks. Nowadays, people usually use Sigmoid function, TANH function and ReLu function as activation functions but you may ask "what's the difference between them and which one should I use?" Don't worry, I will explain them in detail below.

### Sigmoid Function
![Sigmoid graph](/assets/images/Sigmoid.png)
The formula for Sigmoid Function is <code>A = 1/(1+e<sup>-x</sup></code>. It is a non linear function in nature and the output will always in range of (0,1) which is great because it suits binary classification problem. But when z value is either very large or very small, the gradient or the slope of this function becomes very small (close to 0) and this can slow down gradient descent. Next, we are going to talk about TANH function, which is almost always works better than the sigmoid function.

### TANH / Hyperbolic Tangent Function
![TANH graph](/assets/images/TANH.png)
The formular for TANH function is <code>A = e<sup>z</sup>-e<sup>-z</sup>/e<sup>z</sup>+e<sup>-z</sup></code>. The TANH function is a shifted version of the sigmoid function. It's output is in range(-1,1) which is better in most cases than sigmoid function because the mean of the activations are now closer to 0 and the gradient is much stepper. But it still has the same problem that sigmoid function had, when z is extreme large or extreme small, gradient can becomes very small and slow down the gradient descent. So next, I will talk about one other choice that is very popular is called rectify linear unit (ReLu function).

### ReLu Function
![ReLu graph](/assets/images/ReLu.jpeg)
The formular for ReLu function is <code> A = max(0,z)</code>. It's output range is (0,inf), so as long as z is positive, the derivative will be 1 otherwise the derivative will be 0. Imagine a network with random initialized weights and almost 50% of the network yields 0 activation because of the negative value of x which means fewer activation is required and the network is lighter (faster). But ReLu is not perfect, because of the negative value of x, the gradient can go towards 0 which means nothing adjusted during gradient descent. This is called dying ReLu problem. This issue can be solved by making the horizontal line (when x is negative) into non-horizontal component. This is called leaky ReLu. The idea is let the gradient be non zero and recover during training eventually.

### Which one should I use?
If you had to pick one, I would just use the ReLU because there are less gradient that is close to 0 which slows down the gradient descent. I would say never use sigmoid fcuntion unless you are doing binary classification or output layer in neural network, and the reason I said never use it is because TANH function is pretty much superior.

In this article, I tried to explain the pros and cons between three different activation functions. Hope you got a better unstanding of the idea behind activation function, why they are used and how to decide which one to use.
