import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import lassoData

def lasso_regression(X,Y, alpha):
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter = 1e5)
    lassoreg.fit(X, Y)
    return lassoreg.coef_

def ridge_regression(X,Y, alpha):
    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(X,Y)
    return ridgereg.coef_[0]

def SSE(w, X, Y):
    error = 0
    n = X.shape[0]
    for i in xrange(n):
        x = X[i]
        y = Y[i]
        error += (np.dot(w.T, phi(x)) - y)**2
    return error/n


def phi(x):
    v = np.zeros(13)
    v[0] = x
    for i in xrange(1,13):
        v[i] = math.sin(0.4*math.pi*x*i)
    return v

def get_phi_X(X):
    n = X.shape[0]
    phi_X = np.zeros((n,13))
    for i in xrange(n):
        x = X[i]
        phi_X[i] = phi(x)
    return phi_X


def plot_points(X,Y, color="red"):
    if color == "red":
        plt.plot(X, Y, 'ro', label="Training data")
    if color == "blue":
        plt.plot(X, Y, 'bo', label="Test data")
    if color == "green":
        plt.plot(X, Y, 'go', label="Validation data")

def plot_w(w, a):
    xs = np.linspace(-1,1,60)
    ys = []
    for x in xs:
        ys += [np.dot(w.T, phi(x))]
    plt.plot(xs,ys, label = "a = {0}".format(a))
    # plt.plot(xs,ys, label = "a = 10^{0}".format(math.log10(a)))

# rtype = "LASSO"
rtype = "Ridge Regression"

X,Y = lassoData.lassoTrainData()
phi_X = get_phi_X(X)

plt.figure(1)
plt.title("{} with Varying Regularization".format(rtype))
plot_points(X,Y)

possibles = []
l = np.linspace(-3.5,1,20)
alphas = [ 10**i for i in l]
alphas += [0]
for a in alphas:
    w = 0
    if rtype == "LASSO":
        w = lasso_regression(phi_X, Y, a)
    elif rtype == "Ridge Regression":
        w = ridge_regression(phi_X, Y, a)
    possibles += [(a,w)]
    plot_w(w,a)

X_val,Y_val = lassoData.lassoValData()
best_SSE = 10**6
for possible in possibles:
    a,w = possible
    sse_to_train = SSE(w, X, Y)
    sse_to_val = SSE(w, X_val, Y_val)

    if sse_to_val < best_SSE:
        best_SSE = sse_to_val
        best_a = a
        best_w = w

print "Best A: ", best_a
print "Best W: ", best_w
print "Validation Error: ", best_SSE

plt.figure(2)
plt.title("{}".format(rtype))
X_test, Y_test = lassoData.lassoTestData()
plot_points(X, Y)
plot_points(X_val, Y_val, color="green")
plot_points(X_test, Y_test, color="blue")
plot_w(best_w, best_a)
sse_to_test = SSE(best_w, X_test, Y_test)
print "Test Error: ", sse_to_test






plt.legend(loc=2)
plt.show()