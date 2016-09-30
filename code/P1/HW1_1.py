import numpy as np
import math
import matplotlib.pyplot as plt

import loadParametersP1

def gradient_descent(obj_func, gradient_func, init, step_size, threshold):
    theta = init
    gradient = gradient_func(init)
    gradient_norms = []
    print theta, gradient
    counter = 0
    while np.linalg.norm(gradient) > threshold:
        counter += 1
        print counter
        gradient_norms += [np.linalg.norm(gradient)]
        print np.linalg.norm(gradient)
        gradient = gradient_func(theta)
        theta += -1 * step_size * gradient
    print "Minimum at", theta
    print "Value is", obj_func(theta)


    plt.figure(1)
    plt.title("Gradient Diverging")

    plt.plot(gradient_norms)
    plt.show()

def quad_bowl_factory(A, b):
    def quad_bowl(x):
        return (0.5)*np.dot(np.dot(np.transpose(x), A), x) - np.dot(np.transpose(x), b)
    return quad_bowl

def quad_bowl_gradient_factory(A,b):
    def qual_bowl_gradient(x):
        return np.dot(A,x) - b
    return qual_bowl_gradient

def neg_gauss_factory(mean, cov):
    def neg_gauss(x):
        n = len(x)
        exp = -(.5)*np.dot(np.transpose(x-mean), np.dot((np.linalg.inv(cov)),(x-mean)))
        const = -1.0/math.sqrt(((2*math.pi)**n)*np.linalg.det(cov))
        return const*np.exp(exp)
    return neg_gauss

def neg_gauss_gradient_factory(mean, cov):
    def neg_gauss_gradient(x):
        f = neg_gauss_factory(mean,cov)
        return -f(x) * np.dot(np.linalg.inv(cov),x-mean)
    return neg_gauss_gradient

if __name__ == "__main__":
    (gaussMean,gaussCov,quadBowlA,quadBowlb) = loadParametersP1.getData()


    obj_func = quad_bowl_factory(quadBowlA, quadBowlb)
    gradient_func = quad_bowl_gradient_factory(quadBowlA, quadBowlb)
    init = np.zeros(2)
    step_size = 0.15*10**0
    threshold = 10**-5
    print gradient_descent(obj_func, gradient_func, init, step_size, threshold)

    # obj_func = neg_gauss_factory(gaussMean, gaussCov)
    # gradient_func = neg_gauss_gradient_factory(gaussMean, gaussCov)

    # init = np.zeros(2)
    # step_size = 10**8
    # threshold = 10**-10
    # print gradient_descent(obj_func, gradient_func, init, step_size, threshold)