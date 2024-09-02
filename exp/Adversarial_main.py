from functions.Adversarial_functions import *

if __name__ == "__main__":
	num_classes = 10
	input_shape = (28, 28, 1)
	
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
	
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255
	
	x_train = np.expand_dims(x_train, -1)
	x_test = np.expand_dims(x_test, -1)
	print("x_train shape:", x_train.shape)
	print(x_train.shape[0], "train samples")
	print(x_test.shape[0], "test samples")
	
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = keras.Sequential([
	        keras.Input(shape=input_shape),
	        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
	        layers.MaxPooling2D(pool_size=(2, 2)),
	        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
	        layers.MaxPooling2D(pool_size=(2, 2)),
	        layers.Flatten(),
	        layers.Dropout(0.5),
	        layers.Dense(num_classes),
	        layers.Activation("softmax"),
	    ])
	
	model.summary()

	batch_size = 64
	epochs = 10
	
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

	score = model.evaluate(x_train, y_train, verbose=0)
	print("Test loss:", score[0])
	print("Test accuracy:", score[1])

	predictor = keras.Sequential()
	for l in model.layers[:-1]:
	  predictor.add(l)
	predictor.build(input_shape=(None, 28, 28, 1))

	ks = [5*(i+1) for i in range(20)]
	score_true = model.evaluate(x_train, y_train, verbose = 0)[1]
	normal_target = len(np.where(np.argmax(model.predict(x_train, verbose = 0), axis = 1) == target)[0])
	
	score_pert = []
	pert_target = []
	
	perturbation_history = []
	
	w = np.zeros((28,28,1))
	
	for K in ks:
	  max_num_iter = 100
	  c = 2
	  k = 0
	  target = 5
	  batch_size = 1000
	
	  perturbation, f_history, f_history_total = PIHT_CW(w, x_train, predictor, c, k, target, batch_size, max_num_iter, K, eta1 = 0.001, eta2 = 0.001, delta0 = 1, deltamax = 10, gamma = 2)
	
	  perturbation_history.append(perturbation)
	
	  x_train_perturbed = np.floor(transform(transform_back(x_train) + perturbation)*255)/255
	
	  score_pert.append(model.evaluate(x_train_perturbed , y_train, verbose = 0)[1])
	
	  pert_target.append(len(np.where(np.argmax(model.predict(x_train_perturbed , verbose = 0), axis = 1) == target)[0]))
	
	  print(f'\tTarget: {target}\tK: {K}')
	  print('=====================================')
	  print('\t\tnormal\tperturbed')
	  print(f'Accuracy\t{round(score_true, 3)}\t{round(score_pert[-1], 3)}\t')
	  print(f'Target class \t{normal_target}\t{pert_target[-1]}\t')
	  print('=====================================')