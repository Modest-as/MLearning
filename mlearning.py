import numpy
import algebra.linregression as algebra

features = numpy.matrix(
    [
        [0, 1],
        [1, 1],
        [2, 1],
        [3, 1]
    ]
)

coefficients = numpy.matrix([2, 2])
output = numpy.matrix([4, 7, 7, 8])
learning_rate = numpy.diag([0.4, 0.5])

print algebra.cost_function(features, coefficients, output)

# Learning
print "Start learning"

print coefficients

for i in range(0, 10):
    coefficients = algebra.grad_descent_get_next(features, coefficients, output, learning_rate)
    print coefficients

print "End learning"


features = numpy.matrix(
    [
        [1, 1, 1],
        [1, 2, 1],
        [2, 1, 1]
    ]
)

coefficients = numpy.matrix([1, 2, 3])

output = numpy.matrix([3, 4, 4])

print algebra.cost_function(features, coefficients, output)

# Learning
print "Start learning"

print coefficients

for i in range (0, 4200):
    coefficients = algebra.grad_descent_get_next(features, coefficients, output, 0.2)
    print coefficients

print "End learning"
