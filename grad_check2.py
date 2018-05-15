import numpy as np 
from rnn2 import *

np.random.seed(0)
X_length = 3
seq_length = 16
y_length = 3
num_units = 10
batch_size = 20
X_seq = np.random.normal(size=[seq_length,batch_size,X_length])
y_seq = np.random.normal(size=[seq_length,batch_size,y_length])
net = rnn(num_units, X_length, y_length)
net.grad_check(X_seq, y_seq)
