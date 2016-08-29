import numpy


# calculate Sigmoid function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


# calculate the derivative of Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# calculate hypothesis function
# F(C,X) = sigmoid(X * C)
def hypothesis(features, coefficients):
    return sigmoid(features.dot(coefficients.transpose()))


# calculate the cost function for
# J(C) = 1/m * ((-Y ^ TRANSPOSE) * log(h) - (1 - Y) ^ TRANSPOSE * log(1 - h))
# where h = sigmoid(X * C)
# C - vector of coefficients
# X - matrix of features
def cost_function(features, coefficients, output):
    # term that is used when output_(i) = 1
    result = -output.dot(hypothesis(features, coefficients))

    # term that is used when output_(i) = 0
    result -= (1 - output).dot(numpy.log(1 - hypothesis(features, coefficients)))
    return float(result / output.size)


# get coefficient closer to the minimum
# C := C - 1/m * a * (X ^ TRANSPOSE) * (sigmoid(X * C) - Y)
# note that this algorithm is very similar to the one for linear regression
# this is due to the cost function that we have chosen
def grad_descent_get_next(features, coefficients, output, learning_rate):
    # convert to diagonal matrix if learning rate is scalar
    if type(learning_rate) is float or type(learning_rate) is int:
        learning_rate = numpy.identity(coefficients.size) * float(learning_rate)

    diff = hypothesis(features, coefficients) - output.transpose()

    # gradient = 1/m * (X ^ TRANSPOSE) * (sigmoid(X * C)- Y)
    grad = features.transpose().dot(diff) / float(output.size)

    return coefficients - (learning_rate.dot(grad)).transpose()
