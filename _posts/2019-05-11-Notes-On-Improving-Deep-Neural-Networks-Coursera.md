---
title: "Notes on Improving Deep Neural NetWorks Coursera"
date: 2019-05-11 11:41:30
tags: 
  - Machine Learning
  - Deep Learning
  - Neural Networks
categories: "AI" 
mathjax: true
---
# Regularization
If your neural network is overfitting your data means you have a high variance problem. One of the things you can try is regularization. 

## L1 Regularization
Formula on logistic regression is $$\lambda/{2m} \sum_{j=1}^{nx} |W_j|$$. There will be more 0 in w vector. Some people think this can help to compress model because there are more 0s so less memoery required to save model. 

## L2 Regularization (Weight Decay)
Formula on logistic regression is $$\lambda/{2m} \|W\|_2^2 = \sum_{j=1}^{nx} W_j^2 = W^T W$$. 

Formula on neural networks's cost function is $$\lambda/{2m} \sum_{l=1}^L \|W^{[l]}\|_F^2$$. 

Frobenius norm is $$\|W^{[l]}\|_F^2 = \sum_{i=1}^{n^[l-1]} \sum_{j=1}^{n^[l]} (W_{ij}^{[l]})^2$$ Dimension on W is $$(n^{[l]},n^{n[l-1]})$$.

Regularization on back propogation is $$\lambda/m W^{[l]}$$.

## Dropout Regularization 
**Do not use it in test time**
Dropout regularization is random cross out some nodes in layer so we get a smaller nerual network.

Implementing dropout("Inverted dropout") with 3 layer.
```python
keep_prob = 0.8
d3 = np.random.rand(a3.shape[0],a3.shape[1])< keep_prob
a3 = np.multiply(a3,d3)
a3 /= keep_prob
```
a3 /= keep_prob is making sure dropout does not change the expected value in $$a^{[3]}$$ when calculating $$z^{[4]} = W^{[4]} * a^{[3]} + b^{[4]}$$.

# Initialization
## Zero Initialization
If you initialize the weights in nerual networks to zeros then it fails to "break symmetry" which means every neuron in each layer will learn the same thing and you might as well be training a neural network with  n[l]=1 for every layer, and the network is no more powerful than a linear classifier such as logistic regression.

## Random Initialization
To break symmetry, we can init weights randomly.
```python
parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*10
parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
```
## He Initialization
If we have a deep neural network, there will be vanishing / exploding gradients problem because weights is going too big or too small when going thourgh each layer. 
$$ z = W_1 X_1 + W_2 X_2 + W_3 X_3 + \cdots + W_n X_n$$
So we can see the larger the n is, we want the smaller $$W_i$$ to be.
```python
w[i] = parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
```

# Gradient Checking
Gradient checking can help you to know whether there are bugs in your neural networks. There are couple things to remember when implement it.
- Do not use in training - only to debug
- If algorithm fails grad check, look at components to try to identify bug
- Remember regularization
- Does not work with drop out. (do grad check first, then turn on drop out)

Steps to implement:
1. $$\theta^+ = \theta + \epsilon$$
2. $$\theta^- = \theta - \epsilon$$
3. $$J^+ = J(\theta^+)$$
4. $$J^- = J(\theta^-)$$
5. gradapprox = $${J^+ - J^-}/{2\epsilon}$$
6. $$difference = {\|grad - gradapprox\|_2} / {\|grad\|_2 + \|gradapprox\|_2}$$
If the difference is smaller than $$10^{-7}$$ threshold then you have cofidence that you comupted correcly on gradient in backward propagation. 

