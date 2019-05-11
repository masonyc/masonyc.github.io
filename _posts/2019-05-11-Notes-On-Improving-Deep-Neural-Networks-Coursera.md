---
title: "Notes on Improving Deep Neural Networks Coursera"
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
Formula on logistic regression is $$\frac{\lambda}{2m} \sum_{j=1}^{nx} |W_j|$$. There will be more 0 in w vector. Some people think this can help to compress model because there are more 0s so less memoery required to save model. 

## L2 Regularization (Weight Decay)
Formula on logistic regression is $$\frac{\lambda}{2m} \|W\|_2^2 = \sum_{j=1}^{nx} W_j^2 = W^T W$$. 

Formula on neural networks's cost function is $$\frac{\lambda}{2m} \sum_{l=1}^L \|W^{[l]}\|_F^2$$. 

Frobenius norm is $$\|W^{[l]}\|_F^2 = \sum_{i=1}^{n^[l-1]} \sum_{j=1}^{n^[l]} (W_{ij}^{[l]})^2$$ Dimension on W is $$(n^{[l]},n^{n[l-1]})$$.

Regularization on back propogation is $$\frac{\lambda}{m} W^{[l]}$$.

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
1. $\theta^+ = \theta + \epsilon$
2. $\theta^- = \theta - \epsilon$
3. $J^+ = J(\theta^+)$
4. $J^- = J(\theta^-)$
5. gradapprox = $${J^+ - J^-}/{2\epsilon}$$
6. $difference = \frac{\|grad - gradapprox\|_2}{\|grad\|_2 + \|gradapprox\|_2}$

If the difference is smaller than $$10^{-7}$$ threshold then you have cofidence that you comupted correcly on gradient in backward propagation. 

# Optimization Algorithm
## Exponentially Weighted Averages
$$V_t = \beta V_{t-1} + (1-\beta)\theta_t$$ This is averaging over $$\frac{1}{1 - \beta}$$. For example when $$\beta = 0.98$$, we are averging over $$\frac{1}{1-0.98} = 50$$. The bigger the $$\beta$$ you have, the smoother the plot you will get and it will shift more the to right on the graph because you are now averging over a bigger window. It also adapts more slowly since you are averging over a bigger window. The reason is because we give more **weights** to first part of the formula $$\beta V_{t-1}$$ and a much smaller weight to whatever you are seeing rightnow. 

### Bias Correction
Exponentially weighted averages can have bias on the initializatino phase, we can you bias correction to solve the problm
$$\frac{V_t}{1-\beta^t}$$

## Gradient Decent with Momentum
In one sentence, the basic idea is to compute an exponentially weighted average of your gradients, and then use that gradient to update your weights instead. You want slow learning on verticle direction but faster learning on horizontal directoin towards the minimum. After using momentum, you will find out it **averages the verticle direction to 0 as positive and negative value but average value for horizonal directino is still pretty big.**

$V_{dW} = \beta V_{dW} + (1 - \beta)dW$

$V_{db} = \beta V_{db} + (1 - \beta)db$

$W = W - \alpha V_{dW}, b = - \alpha V_{db}$

## RMS prop (Root mean square prop)
We assume **b** is verticle direction and **W** is horizontal direction.

$S_{dW} = \beta S_{dW} + (1 - \beta) dW^2$

$S_{db} = \beta S_{db} + (1 - \beta) db^2$

$W = W - \alpha \frac{dW}{\sqrt{S_{dW}}}$

$b = b - \alpha \frac{db}{\sqrt{S_{db}}}$

We want $$S_{dW}$$ to be small and $$S_{db}$$ to be large so it can slow down the updates on the verticle direction. We can use a large **learning rate alpha** and get faster learning without diverging in the verticle direction. 

## Adam Optimization

$V_{dW} = \beta_1 V_{dW} + (1 - \beta_1) dW$

$V_{db} = \beta_1 V_{db} + (1 - \beta_1) db$

$S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) dW$

$S_{db} = \beta_2 S_{db} + (1 - \beta_2) db$

$V_{dW}^{Corrected} = \frac{V_{dW}}{(1 - \beta_1^t)}$

$V_{db}^{Corrected} = \frac{V_{db}}{(1 - \beta_1^t)}$

$S_{dW}^{Corrected} = \frac{S_{dW}}{(1 - \beta_2^t)}$

$S_{db}^{Corrected} = \frac{S_{db}}{(1 - \beta_2^t)}$

$W = W - \alpha \frac{V_{dW}^{Corrected}}{\sqrt{S_{dW}^{Corrected}}+\epsilon}$

$b = b - \alpha \frac{V_{db}^{Corrected}}{\sqrt{S_{db}^{Corrected}}+\epsilon}$

The default value for $$\beta_1 = 0.9, \beta_2 = 0.99, \epsilon = 10^{-8}$$

## Learning rate decay

$\alpha = \frac{1}{1 - decay_rate * epoch_num} \alpha_0$

$\alpha = 0.95^{epoch_num} \alpha_0$ --Exponentially decay

$\alpha = \frac{k}{\sqrt{epoch_num}} \alpha_0$

# Batch Normalization
$\mu = \frac{1}{m} \sum_{i} z^{(i)}$

$\sigma^2 = \frac{1}{m} \sum_{i} (z_i - \mu)^2$

$ z_{norm}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$ range 0 - 1, mean 0, variance 1

$ \widetilde{z}^{(i)} = \gamma * z_{norm}^{(i)} + \beta$

$$\gamma$$ and $$\beta$$ allows you to set the mean and variance in hidden layer because we dont always want mean = 0 and variance =1.
- Each mini-batch is scaled by the mean / variance computed on just that mini-batch.
- This adds some noise to the values $$z^{[l]}$$ within that minibatch. So similar to dropout, it adds some noise to each hidden layer's activations.
- This has a slight regularization effect.
- A strange property of dropout which is that by using a bigger mini batch size will reduce regularization effect.

# Softmax Regression
Activation function: 
$t = e^{([l])}$

$a^{[l]} = \frac{e^{z^{[l]}}}{\sum_{j=1}^{4} t_i}$

$a^{[l]} = \frac{t_i}{\sum_{j=1}^{4} t_i}$

