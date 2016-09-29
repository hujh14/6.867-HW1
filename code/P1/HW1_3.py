import numpy as np

import loadFittingDataP1
import HW1_1

def least_square_error_factory(X,y):
    def least_square_error(theta):
        return (np.linalg.norm(np.dot(X,theta)-y))**2
    return least_square_error

def LSE_batch_gradient_factory(X, y):
    def least_square_error_batch_gradient(theta):
        return 2*np.dot(X.T,(np.dot(X,theta) - y))
    return least_square_error_batch_gradient


def LSE_batch_gradient_descent():
    (X,y) = loadFittingDataP1.getData()
    obj_func = least_square_error_factory(X,y)
    gradient_func = LSE_batch_gradient_factory(X,y)
    init = np.zeros(X.shape[1])
    step_size = 10**-6
    threshold = 10**-6

    HW1_1.gradient_descent(obj_func, gradient_func, init, step_size, threshold)

# LSE_batch_gradient_descent()





def LSE_SGD_gradient(theta, x, y):
    return 2*(np.dot(x, theta) - y)*x
def learning_rate_func(t):
    # return 10**-6
    tau = 10**9
    K = .75
    return (tau + t)**(-K)

def LSE_SGD(obj_func, gradient_func, init, learning_rate_func, threshold, X, y):
    n = X.shape[0]
    theta = init
    new_cost = obj_func(theta)
    old_cost = 0
    t = 0
    while abs(new_cost - old_cost) > threshold:
        indicies = np.arange(n)
        np.random.shuffle(indicies)

        print new_cost

        old_cost = new_cost
        for i in indicies:
            x_i = X[i]
            y_i = y[i]
            t += 1
            gradient = gradient_func(theta, x_i, y_i)
            theta -= learning_rate_func(t) * gradient
        
        new_cost = obj_func(theta)


    print "Minimum at", theta
    print "Cost is", new_cost

(X,y) = loadFittingDataP1.getData()
obj_func = least_square_error_factory(X,y)
gradient_func = LSE_SGD_gradient
init = np.random.random(X.shape[1])
learning_rate_func = learning_rate_func
threshold = 10**-6

LSE_SGD(obj_func, gradient_func, init, learning_rate_func, threshold, X, y)
