import numpy
import algebra.logregression as algebra

features = numpy.matrix(
    [
        [0.50, 1],
        [0.75, 1],
        [1.00, 1],
        [1.25, 1],
        [1.50, 1],
        [1.75, 1],
        [1.75, 1],
        [2.00, 1],
        [2.25, 1],
        [2.50, 1],
        [2.75, 1],
        [3.00, 1],
        [3.25, 1],
        [3.50, 1],
        [4.00, 1],
        [4.25, 1],
        [4.50, 1],
        [4.75, 1],
        [5.00, 1],
        [5.50, 1],
    ]
)

coefficients = numpy.matrix([1.3, 3.5])
output = numpy.matrix([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

print algebra.cost_function(features, coefficients, output)

# Learning
print "Start learning"

print coefficients

for i in range(0, 6000):
    coefficients = algebra.grad_descent_get_next(features, coefficients, output, 0.1)
    print coefficients

print "End learning"
