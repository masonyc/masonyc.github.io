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
|---
|:-:|:-:|:-:|:-:
| 1 | 0 | -1 
| 1 | 0 | -1
| 1 | 0 | -1 
|---

Sobel kernel gives more weights on central row.
|1|0|-1|
|2|0|-2|
|1|0|-1|

Scharr kernel
|3|0|-3|
|10|0|-10|
|3|0|-3|

