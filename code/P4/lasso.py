import math
import numpy as np

import lassoData


def obj_func_factory(X,Y,l):
    def obj_function(w):
        n = X.shape[0]
        s = 0
        for i in xrange(n):
            x_i = X[i]
            y_i = Y[i][0]
            phi_x_i = phi(x_i)
            s += (y_i - np.dot(w, phi_x_i))**2


        reg_sum = 0
        for i in w:
            reg_sum += abs(i)
        reg = l*reg_sum

        return (1.0/n)*s + reg
    return obj_function

def phi(x):
    v = np.zeros(13)
    for i in xrange(13):
        v[i] = x*math.sin(0.4*math.pi*x*i)
    return v

def gradient_approx(f, w, D):
    n = w.size
    gradient = np.zeros(n)
    d = D/2.0
    for i in xrange(n):
        w_h = np.copy(w)
        w_h[i] = w_h[i] + d
        w_l = np.copy(w)
        w_l[i] = w_l[i] - d
        f_h = f(w_h)
        f_l = f(w_l)
        slope = (f_h-f_l)/D
        gradient[i] = slope
    return gradient

def gradient_descent(obj_func, init, step_size, threshold):
    w = init
    new_cost = obj_func(w)
    old_cost = 0
    gradient = gradient_approx(obj_func, w, 0.0001)
    while abs(new_cost - old_cost) > threshold:
        old_cost = new_cost
        gradient = gradient_approx(obj_func, w, 0.0001)
        w += -1 * step_size * gradient
        # print gradient
        new_cost = obj_func(w)
        print new_cost
    print "Minimum at", w
    print "Cost is", new_cost

X,Y = lassoData.lassoTrainData()
l = 10**0
obj_func = obj_func_factory(X,Y,l)
init = np.zeros(13)
step_size = 10**-2
threshold = 10**-4

gradient_descent(obj_func, init, step_size, threshold)