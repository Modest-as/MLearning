# Machine learning

## Description

Machine learning is quite a fascinating subfield of computer science so I decided to learn more about it. This repository contains some of the code that I have written in order to introduce myself to various concepts of machine learning. 

Who knows, maybe one day this will become an open source machine learning library... It may not be the [TensorFlow](https://www.tensorflow.org/) but this code should give you a basic understanding of the fundamentals. I will try to comment the code the best I can and give some basic theory behind the concepts in this README file.

<b>NOTE:</b> Readers should be familiar with linear algebra and calculus.

## Multivariable Linear Regression

In statistics, [linear regression](https://en.wikipedia.org/wiki/Linear_regression) is an approach for modeling the relationship between a scalar dependent variable ![y](http://mathurl.com/37tklth.png) and one or more explanatory variables (or independent variables) denoted ![x](http://mathurl.com/hx3x3kw.png)).

In a nutshell this means that for any two given data sets of points ![X](http://mathurl.com/3x83h99.png) such that ![x in X](http://mathurl.com/hpkjhsd.png) and ![Y](http://mathurl.com/zjl2ean.png) such that ![y in Y](http://mathurl.com/q38yrve.png) we are trying to find a relationship ![F](http://mathurl.com/hjhy9dv.png) such that ![F(x) = y](http://mathurl.com/jky8dxe.png) where ![x](http://mathurl.com/hx3x3kw.png) represents the input state and ![y](http://mathurl.com/37tklth.png) represents the output for the corresponding input state ![x](http://mathurl.com/hx3x3kw.png).

In this case we are investigating ![F(x)](http://mathurl.com/h3pj3jl.png) that looks something like this:

![Linear function](http://mathurl.com/hwbtzt3.png)

i.e. we are trying to find a set of coefficients ![C](http://mathurl.com/gu7gmsk.png) such that ![F(x)](http://mathurl.com/h3pj3jl.png) is as close to ![y](http://mathurl.com/37tklth.png) as possible. Note that ![x_0 = 1](http://mathurl.com/z854a4q.png) and is called the bias term.

### Cost Function

We measure how well ![F(x)](http://mathurl.com/h3pj3jl.png) describes ![y](http://mathurl.com/37tklth.png) using the cost function:

![Cost function](http://mathurl.com/htwqpzo.png)

where ![x^(i)](http://mathurl.com/zrodoyx.png) is the i-th set of inputs (or features), ![y^(i)](http://mathurl.com/jcxnlez.png) is the output for ![x^(i)](http://mathurl.com/zrodoyx.png) and m is the number of training examples. We can write this in a vector form as:

![Cost in vector form](http://mathurl.com/z2kr4co.png)

where ![C](http://mathurl.com/gu7gmsk.png) is a vector representing all coefficients, ![X](http://mathurl.com/3x83h99.png) is the matrix where every row is a vector ![x^(i)](http://mathurl.com/zrodoyx.png) where i is between 1 and m, and ![y](http://mathurl.com/jou3wvx.png) is a vector representing all outputs ![y^(i)](http://mathurl.com/jcxnlez.png).

### Gradient Descent

In order to find the coefficients ![C](http://mathurl.com/gu7gmsk.png) that minimise our cost function ![J(C)](http://mathurl.com/jqgc83n.png) we use the following algorithm:

![Gradient descent 1](http://mathurl.com/jzvj7rz.png)

where alpha is the learning rate. When we substitute our cost function we get:

![Gradient descent 2](http://mathurl.com/zbx3d6o.png)

The idea behind this is that ![C_i](http://mathurl.com/h9pawlp.png) will converge to some vector ![V](http://mathurl.com/3y4u5qh) which will be the best set of coefficients for our relation ![F(x)](http://mathurl.com/h3pj3jl.png) to predict ![y](http://mathurl.com/37tklth.png). We can choose alpha to be a scalar or a diagonal matrix if we want to adjust the learning rate differently for individual coefficients.

## Logistic Regression

Instead of our output vector ![\vec{y}](http://mathurl.com/jou3wvx.png) having components that are in a continuous range of values, they will be 0 or 1.

We use sigmoid function as our hypothesis representation

![F_C(\vec{x}) = g(X \cdot C) = \frac {1}{1 - e^{-X \cdot C}}](http://mathurl.com/j34dn67.png)

![y = \begin{cases} 1 \text{if } F_C(\vec{x}) \geq 0.5 \\ 0 \text{if }  F_C(\vec{x}) \less 0.5 \end{cases}](http://mathurl.com/hkz5czh.png)

### Cost Function for Logistic Regression

Cost function is:

![J(C) = \frac{1}{m}(-\vec{y}^T \cdot \log{(g(X\cdot C))} - (1 - \vec{y})^T \cdot \log{(1 - g(X\cdot C))})](http://mathurl.com/jt9yln5.png)

This cost function is chosen because it approaches infinity if ![g(X\cdot C) = 1](http://mathurl.com/zgaeq9e.png) while ![y = 0](http://mathurl.com/3snovbu.png) and vice versa. Also, the gradient of this function looks much like the gradient of the cost function for the linear regression.

### Gradient Descent

Using the cost function above our algorithm to find the coefficients looks like this:

![C_{i + 1} = C_i - \alpha \frac{1}{m} X^T \cdot (g(X \cdot C) - \vec{y})](http://mathurl.com/gpwemn7.png)

### References

* Coursera [machine learning](https://www.coursera.org/learn/machine-learning) course
* [Deep Learning](http://www.deeplearningbook.org/) book by Ian Goodfellow Yoshua Bengio and Aaron Courville


 


