from functions.SGGM_functions import *
import GEOparse
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
	

	gds = GEOparse.get_GEO("GDS2910", destdir="./")
	print(gds.table.values.shape)
	
	expression_matrix = gds.table.iloc[:, 2:].values
	expression_matrix = pd.DataFrame(expression_matrix).dropna()
	scaler = StandardScaler()
	expression_matrix = scaler.fit_transform(expression_matrix)
	expression_matrix = np.swapaxes(expression_matrix, 0, 1)

	random.seed(42)
	np.random.seed(42)
	
	indices = np.arange(expression_matrix.shape[0])
	train_indices = np.random.choice(indices, size=150, replace=False)
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