# SimpleNNs

Neural Networks:
	We implemented a feed forward class with one hidden layer.
	The hidden layer size can be specified in the class instantiation.
	To keep think simple, the train function is running over the whole data set.
	So there is no batch decomposition in this version. As a result however, the
	the code needs time to train and might jump from one local minimum to another 
	instead of constantly, slowly improving.

Data sets:
	MNIST (digits) recognition:
		dataset from sklearn
	SPAM BASE spams detection: 
		https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/

Train/Testdecomposition:
	We use 80% of the data for training and 20% for the testing.
	Training is made one various number of epochs, for instance over 5000 for MNIST and
	2500 for SPAMBASE.

Metrics:
	We are measuring LogLoss and Precision.
	They are many additional metrics that could have been used, like recall, f1, confusion matrix, ROC AUC, ...

Results on test:
	>90% precision for MNIST
	~60% precision for SPAMBASE

Conclusion of the Experiments:
	For a simple model and one layer, we could achieve good results on image classification!
	The results on SPAMBASE indicates that one should implement a batch training functionality onto the code,
	because it is likely that the optimisation fails to stabilise around a local minimum and keep jumping.
	



