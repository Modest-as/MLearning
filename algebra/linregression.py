import numpy

# we will note tensor X as
# the following function calculates a cost for
# F(C,X) = c_0 * x_0 + c_1 * x_1 + ... + c_n * x_n
def cost_function(features, coefficients, output):
    # coefficients and results are vectors
    # features is a matrix

    # calculate ((features x coefficients -> 1D) - results) -> 1D
    # this is a vector where each coordinate represents the difference between prediction
    # and an actual value
    diff = features.dot(coefficients.transpose()) - output.transpose()

    # calculate cost function
    # (diff ^ TRANSPOSE) x diff -> scalar
    # this will give us the sum of all differences squared
    # sum of all differences squared is omnidirectional
    # divide this by the number of results doubled
    return float(diff.transpose().dot(diff) / (2.0 * output.size))


# gradient descent is calculated using the following rule
# C := C - 1/m * a * (X ^ TRANSPOSE) * (X * C - Y)
# C - vector of coefficients
# X - matrix of features
# Y - real output
# a - learning rate (a can be a diagonal matrix if you want to adjust learning rates individually)
# m - number of training examples
def grad_descent_get_next(features, coefficients, output, learning_rate):
    # convert to diagonal matrix if learning rate is scalar
    if type(learning_rate) is float or type(learning_rate) is int:
        learning_rate = numpy.identity(coefficients.size) * float(learning_rate)

    diff = features.dot(coefficients.transpose()) - output.transpose()

    # gradient = 1/m * (X ^ TRANSPOSE) * (X * C - Y)
    grad = features.transpose().dot(diff) / float(output.size)

    return coefficients - (learning_rate.dot(grad)).transpose()
