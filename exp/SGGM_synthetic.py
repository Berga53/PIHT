from functions.SGGM_functions import *

def generate_synthetic_data(n_samples, n_features, delta = 0.1, K = 5, random_state = 0):

    p0 = K / (2 * n_features)
    rng = np.random.RandomState(random_state)

    precision = delta * np.eye(n_features)

    B = np.zeros((n_features, n_features))
    bandwidth = K // 2

    for i in range(n_features):
        for j in range(max(0, i - bandwidth), min(n_features, i + bandwidth + 1)):
            B[i, j] = 0.5 ** abs(i - j)

    precision += B

    cov = np.linalg.inv(precision)
    variances = np.diag(cov)
    normalization_matrix = np.diag(1 / np.sqrt(variances))
    cov = normalization_matrix @ cov @ normalization_matrix

    precision = np.linalg.inv(cov)
    precision[np.where(np.abs(precision) < 10**-2)] = 0

    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    return X, precision
	
if __name__ == "__main__":
	
	n_samples = 50
	n_features = 100
	X, true_precision = generate_synthetic_data(n_samples, n_features)
	
	lambda2 = 10**-2
	batch_size0 = 10
	batch_size_max = n_samples
	max_num_iter = 100
	K = 500
	eta1 = 10**-4
	eta2 = 10**-4
	alpha_s = 0.5
	alpha = 1
	
	start = np.eye(n_features)
	
	precision, f_hist, f_hist_total, acc_history, delta_history = PIHT_SGGM(start, X, lambda2,  batch_size0, batch_size_max, max_num_iter, K, eta1, eta2, delta0 = 1, deltamax = 10, gamma = 2)
	print(SGGM_loss(precision, X, lambda2))
	plt.plot(np.arange(1, len(f_hist)+1), f_hist)
	plt.show()
