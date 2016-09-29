import numpy as np
import math

import loadParametersP1

def gradient_descent(obj_func, gradient_func, init, step_size, threshold):
    theta = init
    gradient = gradient_func(init)
    print theta, gradient
    while np.linalg.norm(gradient) > threshold:
        print theta, gradient
        gradient = gradient_func(theta)
        theta += -1 * step_size * gradient
    print "Minimum at", theta
    print "Value is", obj_func(theta)
    return theta, obj_func(theta)

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


    # obj_func = quad_bowl_factory(quadBowlA, quadBowlb)
    # gradient_func = quad_bowl_gradient_factory(quadBowlA, quadBowlb)
    # init = np.zeros(2)
    # step_size = 10**-2
    # threshold = 10**-5
    # print gradient_descent(obj_func, gradient_func, init, step_size, threshold)

    obj_func = neg_gauss_factory(gaussMean, gaussCov)
    gradient_func = neg_gauss_gradient_factory(gaussMean, gaussCov)

    init = np.zeros(2)
    step_size = 10**7
    threshold = 10**-10
    print gradient_descent(obj_func, gradient_func, init, step_size, threshold)