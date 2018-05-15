import numpy as np 
from rnn import *

'''
Train on the sequence (xsin(x)+6sin(5x))/45 (from 0 to 40)
4000 points
seq_length = 32
'''

def fetch_batch(batch_size, seq_length, X):
	random_ix = np.random.permutation(len(X)-seq_length)
	X_batch = []
	y_batch = []
	for i in random_ix[:batch_size]:
		X_batch.append(np.asarray(X[i:i+seq_length].reshape(-1,1)))
		y_batch.append(X[i+1:i+seq_length+1].reshape(-1,1))
	X_batch, y_batch = np.asarray(X_batch), np.asarray(y_batch)
	X_batch, y_batch = np.transpose(X_batch,(1,0,2)), np.transpose(y_batch,(1,0,2))
	return X_batch, y_batch
train_size = 4000
X_train = np.array([((i/10.)*np.sin(i/10.)+6*np.sin(5*(i/10.)))/48 for i in range(train_size)])
X_test = np.array([(((i+0.5)/10.)*np.sin((i+0.5)/10.) + 6*np.sin(5*((i+0.5)/10.)))/48 for i in range(train_size)])
n_epochs = 10
seq_length = 32
n_units = 16
learning_rate = 0.001
batch_size = 1000
n_batches = int(train_size/batch_size)

network = rnn(n_units=n_units, X_length=1, y_length=1)
# train
for epoch in range(n_epochs):
	for batch in range(n_batches):
		learning_rate*=0.94
		X_batch, y_batch = fetch_batch(batch_size, seq_length, X_train)
		X_test_batch, y_test_batch = fetch_batch(batch_size, seq_length, X_test)
		print("train loss: {}".format(network.loss(X_batch,y_batch)))
		print("test loss : {}".format(network.loss(X_test_batch, y_test_batch)))
		network.fit(X_batch, y_batch, learning_rate)

# 
