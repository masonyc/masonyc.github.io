---
title: "Notes on Convolutional Neural Networks Coursera"
date: 2019-05-20 11:41:30
tags: 
  - Machine Learning
  - Deep Learning
  - Neural Networks
categories: "AI" 
mathjax: true
---
# Detect Edges
Positive is the representation of brightness, and negative value is the representation of darkness.

Normal kernal looks like below.

| 1 | 0 | -1 |
| 1 | 0 | -1 |
| 1 | 0 | -1 |

Sobel kernel gives more weights on central row.

| 1 | 0 | -1 |
| 2 | 0 | -2 |
| 1 | 0 | -1 |

Scharr kernel

| 3 | 0 | -3 |
| 10 | 0 | -10 |
| 3 | 0 | -3 |

# Padding
There are two problems when using convolutional network, first problem is every time you apply convolutional operator, the image shrinks. Second problem is the edge of the image in convolutional network is only be used by one of the output. To solve this issue, we can add paddings to the image.
$$p = \frac{f-1}{2}$$

Filter size is $$ f * f $$
Input size is $$ n * n $$
Ouput size is $$ (n + 2p  - f + 1) * (n + 2p  - f + 1)$$ p is padding size

f is usually odd.

# Strided
Strided means how many steps you take (vertically and horizontaly) when applying filter.

Output with stride is $$\lfloor \frac{n + 2p - f}{s} + 1 \rfloor * \lfloor \frac{n + 2p - f}{s} + 1 \rfloor$$

If fraction is not integer, we floor down the number because we can not calculate the area that not full filled.

# Simple Convolutional Network
Convolutional network can also be used on over three dimensional volumes.

Input size $$ n_{H}^{l-1} * n_{W}^{l-1} * n_{c}^{l-1} $$ 
$$n_{c}$$ is the depth of the network

Filter size $$ f^{l} * f^{l} * n_{c}^{l-1}$$ depth need to match with the input

Weights size $$ f^{l} * f^{l} * n_{c}^{l-1} * n_{c}^{l}$$ 

Bias size $$n_{c}^{l}$$

Output size $$ n_{H}^{l} * n_{W}^{l} * n_{c}^{l} $$

Numbers of parameters on fully connected layer are calculated by $$ m * n_{W}^{l-1} * n_{H}^{l-1} * n_{c}^{l-1} + (1,1,1,m) $$ For example last layer is (3,3,3) and this layer has 100 neurons, then number of parameters are $$ 100 * 3 * 3 * 3 + 100 = 2800 $$

Numbers of parameters on convolutional layer are calculated by $$ ( f^{l} * f^{l} * n_{c}^{l-1} + Bias (1) ) * n_{c}^{l} $$ For example last layer is (3,3,3) and we have 10 filters that are each (5,5), then number of parameters are $$ (5 * 5 * 3 + 1) * 10 = 760 $$

# Pooling
## Max Pooling
Take an input and apply the filter on it with step. Take the largest value in the filtered area on input and put it in output position. It usually used to do feature detection because it will keep the high number if feature is detected in the filtered area. Filter size and steps can not be learned from gradient descent. It is fixed value.

## Averages Pooling
Same as Max Pooling but using averages instead. Nowadays, max pooling is being used more often except the neural network is very deep say 7 x 7 x 1000. We use it to collapse the representation.

# Why Convolution
**Parameter Sharing** a feature detector(such as verticle edge detector) thats useful in one part of the image is probably usefule in another part of the image.

**Sparsity of connections** In each layer, each output is only depends on a small set of numbers (filtered area).

# Residual Networks
## The problem of very deep neural networks
The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the lower layers) to very complex features (at the deeper layers). However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow. More specifically, during gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values).

## Building a Residual Network
In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers:

![skip connection graph](/assets/images/skip_connection_kiank.png)

The image on the left shows the "main path" through the network. The image on the right adds a shortcut to the main path. By stacking these ResNet blocks on top of each other, you can form a very deep network.
We also saw in lecture that having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function. This means that you can stack on additional ResNet blocks with little risk of harming training set performance. (There is also some evidence that the ease of learning an identity function--even more than skip connections helping with vanishing gradients--accounts for ResNets' remarkable performance.)
Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. You are going to implement both of them.

### The identity block
The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say  a[l]
 ) has the same dimension as the output activation (say  a[l+2]
 ). To flesh out the different steps of what happens in a ResNet's identity block, here is an alternative diagram showing the individual steps:
 
![identity block graph](/assets/images/idblock2_kiank.png)

### The convolutional block
The CONV2D layer in the shortcut path is used to resize the input  x to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. (This plays a similar role as the matrix  Ws discussed in lecture.) For example, to reduce the activation dimensions's height and width by a factor of 2, you can use a 1x1 convolution with a stride of 2. The CONV2D layer on the shortcut path does not use any non-linear activation function. Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step.

![convolutional block graph](/assets/images/convblock_kiank.png)
 
