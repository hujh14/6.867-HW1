import numpy as np 
import math
import loadFittingDataP2
import matplotlib.pyplot as plt
import pylab as pl
from numpy.polynomial.polynomial import Polynomial as P

def basis_function(n):

    # return lambda x: math.pow(x,n)
    return lambda x: x**n

def cos_function(n):
    return lambda x: np.cos(np.pi*n*x)

def grad_approx(x,h,obj_func):
    n = len(x)
    gradient = []
    for i in range(n):
        unit_vec = np.zeros(n)
        unit_vec[i] = 1
        df_dxi = (obj_func(x+h*unit_vec)-obj_func(x-h*unit_vec))/(2*h)
        gradient.append(df_dxi)

    return np.array(gradient)

def basis_matrix(M,X):
    basis_functions = []

    # creates basis functions starting at x^0 and up to x^m
    for i in range(M+1):
        basis_functions.append(basis_function(i))
        # basis_functions.append(cos_function(i+1))


    basis_functions = np.asarray(basis_functions)
    basis_functions.shape = (M+1,1)
    apply_funcs = np.vectorize(lambda f,x: f(x))

    return apply_funcs(basis_functions,X).T
    

def max_likelihood_w_vector(X, Y, M):
    design_matrix = basis_matrix(M,X)
    # print design_matrix
    d_matrix_pseudo_inverse = np.linalg.inv(np.dot(design_matrix.T,design_matrix))
    return (np.dot(d_matrix_pseudo_inverse,np.dot(design_matrix.T,Y)))

def SSE_grad_2(Y,X,M,w):
    phi = P(w)
    # difference = Y-phi(X)
    basis_functions = []
    gradient = np.zeros(M)

    # creates basis functions starting at x^0 and up to x^m
    for i in range(M):
        basis_functions.append(basis_function(i))

    basis_functions = np.array(basis_functions)

    for x in X:
        difference = Y[i]-phi(x)
        phi_of_x = [func(x) for func in basis_functions]

        phi_of_x = np.array(phi_of_x)
        grad = -difference*phi_of_x
        gradient+= grad

    
    # phi_of_x = [func(x) for func in basis_functions]
    # phi_of_x = np.array(phi_of_x)
    return np.array(gradient)

def SSE_maker(X,Y):
    def SSE(w):
        phi = P(w)
        difference = Y-phi(X)
        return np.sum(difference*difference)/2.0
    return SSE

def SSE_grad_maker(X,Y,M):
    def SSE_grad_5(w):
        phi = P(w)
        # difference = Y-phi(X)
        basis_functions = []
        gradient = np.zeros(M)

        # creates basis functions starting at x^0 and up to x^m
        for i in range(M):
            basis_functions.append(basis_function(i))

        basis_functions = np.array(basis_functions)
        
        for x in X[0]:
            
            difference = Y[i]-phi(x)

            phi_of_x = [func(x) for func in basis_functions]

            phi_of_x = np.array(phi_of_x)
            grad = -difference*phi_of_x

            gradient+= grad

        
        # phi_of_x = [func(x) for func in basis_functions]
        # phi_of_x = np.array(phi_of_x)
        return np.array(gradient)

    return SSE_grad_5

def SSE_2(X,Y,w):
    phi = P(w)
    difference = Y-phi(X)

    return np.sum(difference*difference)/2.0

def batch_gradient_descent(obj_func,gradient_func,init_guess,step_size,convergence_criterion):
    size = init_guess.shape[0]
    w_old = init_guess
    w_new = np.zeros(size)
    alpha = step_size
    old_cost = obj_func(w_old)
    new_cost = np.inf
    iterations = 0
    converged = False
    while (not converged):

        print iterations
        # print gradient_func(w_old)
        print grad_approx(w_old,.0001,obj_func)
        # w_new = w_old - alpha*gradient_func(w_old)
        w_new = w_old - alpha*grad_approx(w_old,.0001,obj_func)
        new_cost = obj_func(w_new)
        
        if np.absolute(new_cost - old_cost) < convergence_criterion:
            converged = True

        old_cost = new_cost
        w_old = w_new

        
        iterations+=1
        

    print 'minimum occurs at: ', w_new
    print "min val", new_cost
    return w_new

# for ML, x = (1,n) y = (n,1)
#for other x = (1,n) y = (1,n)
X,Y = loadFittingDataP2.getData(False)
# Y.shape = (1,11)
Y.shape = (11,1)
# Y = Y[0]

X.shape = (1,11)

# [[  2.36687552]
#  [-10.73144911]
#  [  6.61648601]
#  [  2.48074981]]
M=5

w = max_likelihood_w_vector(X,Y,M)
print w
# 25-52
# p = P([0.7789928 , 1.17413213])
# f = lambda x: 0.7789928*np.cos(x*np.pi) + 1.17413213*np.cos(x*np.pi*2)
# x = np.linspace(0,1,100)
# y = f(x)
# plt.plot(X,Y,'o')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x,y)
# plt.show()

# sse = SSE_maker(X,Y)
# sse_grad = SSE_grad_maker(X,Y,3)
# # guess = np.array([1,-7,5,1])
# guess = np.random.random(3)

# w = batch_gradient_descent(sse,sse_grad,guess,.0001,10**-3)
w.shape = (1,M+1)
p = P(w[0])
# f = lambda x: 0.7789928*np.cos(x*np.pi) + 1.17413213*np.cos(x*np.pi*2)
x = np.linspace(0,1,100)
y = p(x)
X = X[0]

plt.plot(X,Y,'o')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.show()


# X = np.asarray([2,4,6])
# Y = np.array([1,1,1])
# M = 3
# w = np.array([1,1,1])


# print SSE_grad_2(Y,X,3,w)
# print grad_approx(w,.0001,sse)

# print SSE_2(X,Y,w)


# basis_functions = []

# # creates basis functions starting at x^0 and up to x^m
# for i in range(4):
#   basis_functions.append(basis_function(i))

# basis_functions = np.array(basis_functions)

# # basis_functions.shape = (4,1)

# # print basis_functions
# phi_of_x = [func(3) for func in basis_functions]
# print phi_of_x