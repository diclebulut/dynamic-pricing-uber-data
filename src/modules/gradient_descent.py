import numpy as np
from autograd import value_and_grad

def model(w, x):
    return np.dot(x[0].T, w)

def least_squares(w, x, y):
    cost = np.sum((model(w, x) - y) ** 2)
    return cost

def gradient_descent(g, alpha, max_its, w, x, y):
    gradient = value_and_grad(g)
    weight_history = []
    cost_history = []
    for k in range(1, max_its + 1):
        cost_eval, grad_eval = gradient(w, x, y)
        weight_history.append(w)
        cost_history.append(cost_eval)
        w = w - alpha * grad_eval
    weight_history.append(w)
    cost_history.append(g(w, x, y))
    cost_history = np.asarray(cost_history) / x.shape[2]
    return weight_history, cost_history

def demand_curve(axis, value):
    demand = []
    sample_size = len(value)
    for price in axis:
        demand.append(sum(1 if x >= price else 0 for x in value) / sample_size)
    return demand

def APP_s(axis, value, c):
    demand = demand_curve(axis, value)
    app_s = []
    for i in range(len(axis)):
        app_s.append(max(axis[i] - c, 0) * demand[i])
    return app_s

def APP_d(w, x, y, c, d):
    est = d * model(w, x)
    ind = list(range(len(est)))
    rev = sum(max(est[i], c) if y[0][i] >= max(est[i], c) else 0 for i in ind)
    return rev / x.shape[2]

