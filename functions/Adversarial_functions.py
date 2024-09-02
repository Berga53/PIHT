import numpy as np
import keras
from keras import layers
import tensorflow as tf
import random

def sample(X, batch_size):

    indices = np.arange(len(X))
    batch_indices = np.random.choice(indices, size=batch_size, replace=False)
    X_batch = X[batch_indices]

    return X_batch

def transform(X):

  return 0.5*np.tanh(X)+0.5

def transform_back(X):
  scaled_X = X.copy()
  scaled_X[scaled_X >= 1] = 254.5/255
  scaled_X[scaled_X <= 0] = .5/255

  return np.arctanh(2*scaled_X - 1)

def f(probs, k, target):
    result = []
    if target is None:
        for elem in probs:
            top_2_values = np.partition(elem, -2)[-2:]
            max_value = top_2_values[-1]
            second_max_value = top_2_values[-2]
            difference = max_value - second_max_value
            result.append(np.max([difference, -k]))
    else:
        for i,elem in enumerate(probs):
            result.append(np.max((np.max(np.delete(elem, target) - elem[target]), -k)))
    return np.array(result)

def CW_loss(x, w, model, c, k, target):

  w1 = transform(transform_back(x)+w)

  probs = np.array(model(w1))

  loss1 = np.linalg.norm(w1-x, axis = (1,2)).T

  loss2 = f(probs, k, target)

  return np.sum(loss1+c*loss2)/len(x)

def CW_gradient(x, w, model, c, k, target):

    w1 = transform(transform_back(x)+w)

    probs = np.array(model(w1))

    grad1 = w1/np.linalg.norm(w1-x)

    w_tensor = tf.convert_to_tensor(w1, dtype=tf.float32)

    if target is None:
        with tf.GradientTape() as t:
            t.watch(w_tensor)
            predictions = predictor(w_tensor)
            top_2_values = tf.nn.top_k(predictions, k=2, sorted=True).values
            max_value = top_2_values[:, 0]
            second_max_value = top_2_values[:, 1]
            output = max_value - second_max_value

    else:
        with tf.GradientTape() as t:
            t.watch(w_tensor)
            output = tf.reduce_max(predictor(w_tensor), axis = 1) - predictor(w_tensor)[:,target]

    grad2 = t.gradient(output, w_tensor)

    g_stack = 0.5*np.power(np.cosh(transform_back(x)+w), -2)*(grad1 + c*grad2)

    return np.mean(g_stack, axis = 0)

def HT(x, K):

    tr = -np.sort(-np.abs(x), axis = None)[K-1]
    x[np.abs(x)<tr] = 0

    return x

def PIHT_CW(w0, X, model, c, k, target, batch_size, max_num_iter, K, eta1 = 0.001, eta2 = 0.001, delta0 = 1, deltamax = 10, gamma = 2):

    delta = delta0
    num_iter = 0
    w = [HT(w0, K)]
    f_hist = []
    f_hist_total = []

    while (num_iter < max_num_iter):

        X_batch = sample(X, batch_size)
        gk = CW_gradient(X_batch, w[-1], model, c, k, target)
        wk = HT(w[-1] - min(delta / np.linalg.norm(gk), 1)*gk, K)

        X_batch = sample(X, batch_size)

        f1 = CW_loss(X_batch, w[-1], model, c, k, target)
        fk = CW_loss(X_batch, wk, model, c, k, target)

        if (((f1-fk)/(np.linalg.norm(gk)*delta) > eta1) and (np.linalg.norm(gk) > eta2*delta)):
            delta = min(delta * gamma, deltamax)
            w.append(wk)
            num_iter += 1
            f_hist.append(fk)
            f_hist_total.append(fk)

        else:
            delta = delta / gamma
            num_iter += 1
            w.append(w[-1])
            f_hist_total.append(fk)

        print(f'\r{num_iter}', end=" ")

    return w[-1], f_hist, f_hist_total