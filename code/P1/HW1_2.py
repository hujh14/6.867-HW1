import numpy as np
import HW1_1
import loadParametersP1

# def gradient_approx(f, x, D):
#     d = D/2.0
#     d_x_h = f(np.array([x[0]+d, x[1]]))
#     d_x_l = f(np.array([x[0]-d, x[1]]))
#     d_y_h = f(np.array([x[0], x[1]+d]))
#     d_y_l = f(np.array([x[0], x[1]-d]))
#     d_x = (d_x_h-d_x_l)/D
#     d_y = (d_y_h-d_y_l)/D
#     return np.array([d_x, d_y])

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

def evaluate_quad_bowl_gradient_accuracy(x):
    (gaussMean,gaussCov,quadBowlA,quadBowlb) = loadParametersP1.getData()
    quad_bowl_func = HW1_1.quad_bowl_factory(quadBowlA, quadBowlb)
    quad_bowl_grad_func = HW1_1.quad_bowl_gradient_factory(quadBowlA, quadBowlb)
    print "Actual value: ", quad_bowl_grad_func(x)
    ds = [100,7,5,3,.9,.5,.3,.1]
    for d in ds:
        gradient = gradient_approx(gaussian_func, x, d)
        print d, gradient

def evaluate_gaussian_gradient_accuracy(x):
    (gaussMean,gaussCov,quadBowlA,quadBowlb) = loadParametersP1.getData()
    gaussian_func = HW1_1.neg_gauss_factory(gaussMean, gaussCov)
    gaussian_grad_func = HW1_1.neg_gauss_gradient_factory(gaussMean, gaussCov)
    print "Actual value: ", gaussian_grad_func(x)
    ds = [100,7,5,3,.9,.5,.3,.1]
    for d in ds:
        gradient = gradient_approx(gaussian_func, x, d)
        print d, gradient



x = np.array([11.0,11.0])
# evaluate_quad_bowl_gradient_accuracy(x)
evaluate_gaussian_gradient_accuracy(x)


