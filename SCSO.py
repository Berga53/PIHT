import numpy as np
import pandas as pd
import random
import pickle

def sample(X, batch_size):

    indices = np.arange(X.shape[0])
    batch_indices = np.random.choice(indices, size=batch_size, replace=False)
    X_batch = X[batch_indices,:]

    return X_batch

def SGGM_loss(theta, X, lambda2):

    n = X.shape[0]

    Xt = X/(n**0.5)

    loss1 = - np.sum(np.log(np.diag(theta)))

    loss2 = 0

    for i in range(theta.shape[0]):
        loss2 += (1/theta[i,i]) * np.linalg.norm(Xt @ theta[:,i]) ** 2


    loss3 = lambda2 * np.linalg.norm(theta)**2

    return loss1 + loss2 + loss3

def SGGM_gradient(theta, X, lambda2):

    n = X.shape[0]

    Xt = X/(n**0.5)
    # Xt = X/n

    X_2 = Xt.T @ Xt

    grad1 = - (np.diag(theta)**-1)

    grad2 = []
    grad3 = []

    for i in range(theta.shape[0]):
        grad2.append((-1 / (theta[i,i]) ** 2) * np.linalg.norm(Xt @ theta[:,i]) ** 2)
        grad3.append(2 / theta[i,i] * (X_2 @ theta[:,i]))

    grad2 = np.array(grad2)
    grad3 = np.array(grad3)

    grad4 = 2 * lambda2 * theta

    grad = np.diag(grad1 + grad2) + grad3 + grad4

    return np.tril(grad) + np.tril(grad).T - np.diag(np.diag(grad))

def HT(x, K):

    tr = -np.sort(-np.abs(x), axis = None)[K-1]
    x[np.abs(x)<tr] = 0

    return x

def batch_size_f(delta, delta0, batch_size_max, batch_size0):
    if delta == 0.0:
        return batch_size_max

    elif (delta0/delta) > 12:
        return batch_size_max

    else:
        return min(batch_size_max, int(batch_size0 * 2**(delta0/delta)))

def PIHT_SGGM(w0, X, lambda2, batch_size0, batch_size_max, max_num_iter, K, eta1 = 0.001, eta2 = 0.001, delta0 = 1, deltamax = 10, gamma = 2):

    delta = delta0
    num_iter = 0
    w = [HT(w0, K)]
    f_hist = []
    f_hist_total = []
    acc_history = []
    delta_history = []

    while (num_iter < max_num_iter):

        delta_history.append(delta)
        batch_size = batch_size_f(delta, delta0, batch_size_max, batch_size0)
        X_batch = sample(X, batch_size)
        gk = SGGM_gradient(w[-1], X_batch, lambda2)
        wk = HT(w[-1] - min(delta / np.linalg.norm(gk), 1)*gk, K)

        X_batch = sample(X, batch_size)

        f1 = SGGM_loss(w[-1], X_batch, lambda2)
        fk = SGGM_loss(wk, X_batch, lambda2)

        if (((f1-fk)/(np.linalg.norm(gk)*delta) > eta1) and (np.linalg.norm(gk) > eta2*delta)):
            delta = min(delta * gamma, deltamax)
            w.append(wk)
            num_iter += 1
            f_hist.append(fk)
            f_hist_total.append(fk)
            acc_history.append(1)

        else:
            delta = delta / gamma
            num_iter += 1
            w.append(w[-1])
            f_hist_total.append(fk)
            acc_history.append(0)

    return w[-1], f_hist, f_hist_total, acc_history, delta_history

def generate_symmetric_matrix_with_k_nonzeros(n, k):
    if k < n or k > n * (n + 1) // 2:
        raise ValueError("k must be greater than or equal to n and less than or equal to n*(n+1)/2")

    matrix = np.eye(n, dtype=float)
    num_off_diagonal_ones = k - n
    possible_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    selected_pairs = random.sample(possible_pairs, num_off_diagonal_ones)

    for (i, j) in selected_pairs:
        matrix[i, j] = 0.2
        matrix[j, i] = 0.2

    return matrix

if __name__ == "__main__":
	
	path = 'GDS2910.txt' #Insert your path here
	
	expression_matrix = np.loadtxt(path)

	random.seed(42)
	np.random.seed(42)
	
	indices = np.arange(expression_matrix.shape[0])
	train_indices = np.random.choice(indices, size=150, replace=False) #150/191 should be enough, in case tell me so I change it also in my experiments
	val_indices = [idx for idx in indices if idx not in train_indices]
	X = expression_matrix[train_indices,:]
	val_X = expression_matrix[val_indices,:]
	print(X.shape, val_X.shape)

	n_features = X.shape[-1]
	batch_size0 = 50
	batch_size_max = X.shape[0]
	max_num_iter = 1000
	lambda2 = 0
	
	eta1 = 10**-4
	eta2 = 10**-4
	
	Ks = np.arange(5000,15500,500)
	
	for K in Ks:
		
	    temp = []
	    for _ in range(10):
			
	        start = generate_symmetric_matrix_with_k_nonzeros(n_features, K)
	        print(SGGM_loss(start, X, lambda2))
	        precision, f_hist, f_hist_total, acc_history, delta_history = PIHT_SGGM(start, X, lambda2, batch_size0, batch_size_max, max_num_iter, K, eta1, eta2, delta0 = 1, deltamax = 10, gamma = 2)
	        print(SGGM_loss(precision, X, lambda2))
	        temp.append((precision, f_hist, f_hist_total, acc_history, delta_history))
	        
	    
	    with open(f'output_{K}.pkl', 'wb') as f:
	        pickle.dump(temp, f)