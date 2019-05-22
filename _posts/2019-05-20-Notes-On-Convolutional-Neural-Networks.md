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

Numbers of parameters on fully connected layer are calculated by $$ m * n_{W}^{l-1} * n_{H}^{l-1} * n_{c}^{l-1} + (1,1,1,m) $$ For example last layer is (3,3,3) and this layer has 100 neurons, then number of parameters are $$100*3*3*3+100 = 2800$$

Numbers of parameters on convolutional layer are calculated by $$ (f^{l} * f^{l} * n_{c}^{l-1} + Bias (1) ) * n_{c}^{l} $$ For example last layer is (3,3,3) and we have 10 filters that are each (5,5), then number of parameters are $$ (5 * 5 * 3 + 1) * 10 = 760 $$

# Pooling
## Max Pooling
Take an input and apply the filter on it with step. Take the largest value in the filtered area on input and put it in output position. It usually used to do feature detection because it will keep the high number if feature is detected in the filtered area. Filter size and steps can not be learned from gradient descent. It is fixed value.

## Averages Pooling
Same as Max Pooling but using averages instead. Nowadays, max pooling is being used more often except the neural network is very deep say 7 x 7 x 1000. We use it to collapse the representation.

# Why Convolution
**Parameter Sharing** a feature detector(such as verticle edge detector) thats useful in one part of the image is probably usefule in another part of the image.

**Sparsity of connections** In each layer, each output is only depends on a small set of numbers (filtered area).

