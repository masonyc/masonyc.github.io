---
title: "K Nearest Neighbors Algorithm"
date: 2019-03-31 19:56:20
tags: 
  - Machine Learning
  - Machine Learning In Action 
categories: "Machine_Learning" 
---
Today I am going to talk about an algorithm called 'K Nearest Neighbors Algorithm'. Assume we have an exisintg dataset and we labeled all the data in the dataset - we know what class each piece of data should fall into. When we have a new piece of data without a label, we calculate the distance of the new data to all the data in our dataset, then we take the k-closest(neighbors) datas and look at their labels. Lastly, we take the label that have most apperance in the k-closest datas and classify the new data as one of them.

Lets look at the code snippet that implements the algorithm below.<br/>
<code>
def classify(inX, dataSet, labels, k):<br/>
    ""<br/>"
    inX: input data <br/>
    dataSet: origin dataset <br/>
    labels: origin labels<br/>
    k: number of nearest target<br/>
    """<br/>
    # Calculate Distance between new data and dataset<br/>
    dataSetSize = dataSet.shape[0]  # number of rows in dataSet<br/>
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet<br/>
    sqDiffMat = diffMat ** 2<br/>
    sqDistance = sqDiffMat.sum(axis=1)  # sum  matrix into a vector<br/>
    distance = sqDistance ** 0.5<br/>
    # Calculate finish, sort the result is from nearest to the farthest<br/>
    sortedDistanceIndex = argsort(distance)  # index of sorted distance (small to big)<br/>
    classCount = {}<br/>
    for i in range(k):<br/>
        voteIlabel = labels[sortedDistanceIndex[i]]<br/>
        # add k-closest label to dict<br/>
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1<br/>
    sortedClassCount = sorted(classCount.iteritems(),<br/>
                              key=operator.itemgetter(1), reverse=True)<br/>
    return sortedClassCount[0][0]<br/>
</code> 
The function classify takes four input parameters: input vector to classify called inX, full matrix of training examples called dataSet, a vector of labels called labels, and finally k, the number of nearest data we are looking for.

We use the Euclidian distance to calculate the distance between two vectors,xA and xB, with two elements, is given by 
<code>&#8730;((xA<sub>0</sub>-xB<sub>0</sub>)<sup>2</sup> + (xA<sub>1</sub>-xB<sub>1</sub>)<sup>2</sup>)</code>

Then we sort the distance from nearest to the farthest and take the indices in to an array. Now we go through a for loop from 0 to K and save the label and the appearance in to a dict. Lastly, we return the label that have the most appearance. 

To sum up, KNN algorithm is a simple algorithm that has high accuracy, insensitive to outliers, and no assumptions about data. But it is computationally expensive and requires a lot of memory because it needs to compare the new data to all the data in dataset.
