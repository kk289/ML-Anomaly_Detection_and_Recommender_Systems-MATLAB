# Machine Learning (MATLAB) - Anomaly Detection and Recommender Systems

Machine Learning course from Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/home/week/9).

### Introduction
We will implement the anomaly detection algorithm and apply it to detect failing servers on a network. In the second part, we will use collaborative filtering to build a recommender system for movies.

### Environment
- macOS Catalina (version 10.15.3)
- MATLAB 2018 b

### Dataset
- ex8data1.mat
- ex8data2.mat
- ex8_movies.mat

### Files included in this repo
- ex8.m - Octave/MATLAB script for first part of exercise
- ex8_cofi.m - Octave/MATLAB script for second part of exercise 
- ex8data1.mat - First example Dataset for anomaly detection 
- ex8data2.mat - Second example Dataset for anomaly detection
- ex8_movies.mat - Movie Review Dataset
- ex8_movieParams.mat - Parameters provided for debugging 
- multivariateGaussian.m - Computes the probability density function for a Gaussian distribution
- visualizeFit.m - 2D plot of a Gaussian distribution and a dataset 
- checkCostFunction.m - Gradient checking for collaborative filtering 
- computeNumericalGradient.m - Numerically compute gradients
- fmincg.m - Function minimization routine (similar to fminunc) 
- loadMovieList.m - Loads the list of movies into a cell-array
- movie_ids.txt - List of movies
- normalizeRatings.m - Mean normalization for collaborative filtering 
- submit.m - Submission script that sends code to our servers 

[⋆] estimateGaussian.m - Estimate the parameters of a Gaussian distribution with a diagonal covariance matrix

[⋆] selectThreshold.m - Find a threshold for anomaly detection

[⋆] cofiCostFunc.m - Implement the cost function for collaborative filtering  

## Anomaly Detection
We will implement an anomaly detection algorithm to detect anomalous behavior in server computers. The features measure the through put (mb/s) and latency (ms) of response of each server. While our servers were operating, we collected m = 307 examples of how they were behaving, and thus have an unlabeled dataset {x(1),...,x(m)}. We suspect that the vast majority of these examples are “normal” (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.

We will use a Gaussian model to detect anomalous examples in dataset.
First start on a 2D dataset that will allow to visualize what the algorithm is doing. On that dataset we will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies. After that, apply the anomaly detection algorithm to a larger dataset with many dimensions.

```
ex8.m
```

![dataset1](Figure/dataset1.jpg)
- Figure: Visualize first dataset

### Part 1.1: Gaussian Distribution
To perform anomaly detection, we first need to fit a model to the data's distribution.  
The Gaussian distribution:  
![gaussian](Figure/gaussian.png)    
where, μ is the mean and σ2 controls the variance.







## Course Links 
1) Machine Learning by Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/home/week/9).

2) [Anomaly Detection and Recommender Systems](https://www.coursera.org/learn/machine-learning/home/week/9)
(Please notice that you need to log in to see the programming assignment.) #ML-Anomaly_Detection_and_Recommender_Systems-MATLAB