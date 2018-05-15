
import numpy as np 

class rnn:
	def __init__(self, num_units, X_length, y_length):
		self.num_units = num_units
		self.X_length = X_length
		self.y_length = y_length
		self.Wx = np.random.normal(scale=1.0/np.sqrt(X_length*num_units),size=[X_length, num_units])
		self.bxh = np.zeros([1,num_units])
		self.Wh = np.random.normal(scale=1.0/np.sqrt(num_units*num_units),size=[num_units, num_units])
		self.Wo = np.random.normal(scale=1.0/np.sqrt(num_units*y_length),size=[num_units, y_length])
		self.bo = np.zeros([1,y_length])

	def relu(self,x):
		return x*(x>0)

	def predict(self, X_seq):
		seq_length, batch_size, X_length = X_seq.shape
		if X_length!=self.X_length:
			print("X_length != self.X_length")
			exit()
		o_seq = []
		h_seq = [np.zeros([batch_size,self.num_units])]
		for k in range(seq_length):
			s = np.matmul(h_seq[-1],self.Wh)+np.matmul(X_seq[k],self.Wx)+np.repeat(self.bxh,batch_size,axis=0)
			h = self.relu(s)
			o = np.matmul(h,self.Wo)+np.repeat(self.bo,batch_size,axis=0)
			h_seq.append(h)
			o_seq.append(o)
		h_seq = h_seq[1:]
		o_seq = np.asarray(o_seq)
		h_seq = np.asarray(h_seq)
		return h_seq, o_seq

	def grad(self, X_seq, y_seq):
		h_seq, o_seq = self.predict(X_seq)
		seq_length, batch_size, X_length = X_seq.shape
		
		d_Wo = np.zeros(self.Wo.shape)

		for k in range(seq_length):
			d_o = 1.0/batch_size*(o_seq[k]-y_seq[k])
			d_Wo += np.matmul(np.transpose(h_seq[k]),d_o)


		return d_Wo 



	def loss(self, X_seq, y_seq):
		h_seq, o_seq = self.predict(X_seq)
		seq_length, batch_size, X_length = X_seq.shape
		loss = 1.0/(2*batch_size)*np.sum(np.square(o_seq-y_seq))
		return loss

	def grad_check(self, X_seq, y_seq):
		d_theta = 1e-5
		threshold = 1e-3
		d_Wo = self.grad(X_seq, y_seq)

		for i in range(self.Wo.shape[0]):
			for j in range(self.Wo.shape[1]):
				self.Wo[i,j] += d_theta
				loss_plus = self.loss(X_seq, y_seq)
				self.Wo[i,j] -= 2*d_theta
				loss_minus = self.loss(X_seq, y_seq)
				self.Wo[i,j] += d_theta

				num_grad = (loss_plus-loss_minus)/(2*d_theta)
				if np.absolute((num_grad-d_Wo[i,j])/num_grad) > threshold:
					print("Wrong grad at Wo[{},{}]".format(i,j))
					print("Num grad:", num_grad)
					print("Theoretical grad:", d_Wo[i,j])


