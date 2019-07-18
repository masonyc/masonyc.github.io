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
 
# YOLO Algorithm
YOLO ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

For each grid cell we define y as [px,bx,by,bh,bw,c1,c2,c3], px is the probability to have a object that we might intersted in. bx, by, bh, bw is the bounding box position on the object. **We assign the mid point of the object to the grid cell that contains the mid point.** 

## Non-Max Suppression
Even after filtering by thresholding over the classes scores, you still end up a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 

![Non-max-suppression](/assets/non-max-suppression.png)

Non-max suppression uses the very important function called "Intersection over Union", or IoU.

![Intersection over union](/assets/iou.png)


## Implementation 
Steps to Implement :
- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
- Each cell in a 19x19 grid over the input image gives 425 numbers.
- 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
- 85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw) has 5 numbers, and and 80 is the number of classes we'd like to detect

You then select only few boxes based on:
- Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
- Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives you YOLO's final output.
``` python
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    """
    
    # Step 1: Compute box scores
    box_scores =  box_confidence*box_class_probs
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1,keepdims=False)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores>=threshold
    
    # Step 4: Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores, boxes, classes
    
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.maximum(box1[0],box2[0])
    yi1 = np.maximum(box1[1],box2[1])
    xi2 = np.minimum(box1[2],box2[2])
    yi2 = np.minimum(box1[3],box2[3])
    inter_area = max(yi2-yi1,0) * max(xi2-xi1,0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area+box2_area-inter_area
    
    # compute the IoU
    iou = inter_area/union_area
    
    return iou
    
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor,iou_threshold=iou_threshold)
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores,nms_indices)
    boxes =  K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
    
    return scores, boxes, classes    
```

## 
