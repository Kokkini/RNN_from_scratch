import numpy as np

def relu(x):
    return x*(x>0)

class rnn:
    def __init__(self, n_units, X_length, y_length):
        self.n_units = n_units
        self.Wx = np.random.normal(scale=1/np.sqrt(X_length*n_units),size=[X_length,n_units])
        self.Wh = np.random.normal(scale=1/n_units,size=[n_units,n_units])
        self.Wo = np.random.normal(scale=1/np.sqrt(n_units*y_length),size=[n_units,y_length])
        self.bxh = np.zeros(shape=[1,n_units])
        self.bo = np.zeros(shape=[1,y_length])
    
    def predict(self, X_seq): #X_seq.shape = [seq_length,?,X_length]
        output_seq = []
        h_seq = [np.zeros(shape=[X_seq.shape[1],self.n_units])] # len(h_seq) will be 1 bigger than output_seq
        seq_length = len(X_seq)
        for k in range(seq_length):
            s = np.matmul(X_seq[k],self.Wx)+np.matmul(h_seq[-1],self.Wh)+self.bxh
            h = relu(s)
            o = np.matmul(h,self.Wo)+self.bo
            output_seq.append(o)
            h_seq.append(h)
        h_seq = h_seq[1:]
        h_seq = np.asarray(h_seq)
        output_seq = np.asarray(output_seq)
        return output_seq, h_seq
    
    def grad(self, X_seq, y_seq):
        seq_length, batch_size, X_length = X_seq.shape
        output_seq, h_seq = self.predict(X_seq)
        
        d_Wx = np.zeros(self.Wx.shape)
        d_Wh = np.zeros(self.Wh.shape)
        d_Wo = np.zeros(self.Wo.shape)
        d_bxh = np.zeros(self.bxh.shape)
        d_bo = np.zeros(self.bo.shape)
        
        # d_o = np.zeros([batch_size, self.y_length])
        # d_h = np.zeros([batch_size, self.n_units])
        # d_s = np.zeros([batch_size, self.n_units])
        
        for k in range(seq_length):
            d_o = (1/batch_size) * (output_seq[k]-y_seq[k])
            d_Wo += np.matmul(np.transpose(h_seq[k]),d_o)
            d_bo += np.sum(d_o,axis=0)
            
            d_h = np.matmul(d_o, np.transpose(self.Wo))
            for i in range(k):
                d_s = d_h*(h_seq[k-i]>0)
                d_Wx += np.matmul(np.transpose(X_seq[k-i]),d_s)
                d_bxh += np.sum(d_s, axis=0)
                d_Wh += np.matmul(np.transpose(h_seq[k-i-1]),d_s)
                d_h = np.matmul(d_s,np.transpose(self.Wh))
            # the first instance in the seq is dealt with separately because it involves the 0 state
            d_s = d_h*(h_seq[0]>0)
            d_Wx += np.matmul(np.transpose(X_seq[0]),d_s)
            d_bxh += np.sum(d_s,axis=0)
        return d_Wx, d_Wh, d_Wo, d_bo, d_bxh 

    def fit(self, X_seq, y_seq, learning_rate): #X_seq.shape = [seq_length,?,X_length]
        d_Wx, d_Wh, d_Wo, d_bo, d_bxh = self.grad(X_seq, y_seq)
        # update
        self.Wx -= learning_rate*d_Wx
        self.Wh -= learning_rate*d_Wh
        self.Wo -= learning_rate*d_Wo
        self.bo -= learning_rate*d_bo
        self.bxh -= learning_rate*d_bxh

    def loss(self, X_seq, y_seq):
        seq_length, batch_size, X_length = X_seq.shape
        output_seq, h_seq = self.predict(X_seq)
        loss = 1/(2*batch_size)*np.sum(np.square(output_seq-y_seq))
        return loss

    def grad_check(self, X_seq, y_seq):
        d_theta = 1e-5
        threshold = 1e-3
        d_Wx, d_Wh, d_Wo, d_bo, d_bxh = self.grad(X_seq, y_seq)
        e_Wx, e_Wh, e_Wo, e_bo, e_bxh = np.zeros(self.Wx.shape), np.zeros(self.Wh.shape), np.zeros(self.Wo.shape), np.zeros(self.bo.shape), np.zeros(self.bxh.shape)
        # grad of Wx
        for i in range(e_Wx.shape[0]):
            for j in range(e_Wx.shape[1]):
                self.Wx[i,j] += d_theta
                loss_plus = self.loss(X_seq, y_seq)
                self.Wx[i,j] += -2*d_theta
                loss_minus = self.loss(X_seq, y_seq)
                self.Wx[i,j] += d_theta
                num_grad = (loss_plus-loss_minus)/(2*d_theta)
                error = np.absolute(d_Wx[i,j]-num_grad)
                e_Wx[i,j] = error
                if error/np.absolute(num_grad)>threshold:
                    print("Wrong grad at Wx[{},{}]".format(i,j))
                    print("Numerical grad:  ",num_grad)
                    print("Theoretical grad:",d_Wx[i,j])
        print("\n\n\n")
        # grad check for Wh
        for i in range(self.Wh.shape[0]):
            for j in range(self.Wh.shape[1]):
                self.Wh[i,j] += d_theta
                loss_plus = self.loss(X_seq,y_seq)
                self.Wh[i,j] -= 2*d_theta
                loss_minus = self.loss(X_seq,y_seq)
                self.Wh[i,j] += d_theta
                num_grad = (loss_plus-loss_minus)/(2*d_theta)
                error = np.absolute(d_Wh[i,j]-num_grad)
                e_Wh[i,j] = error
                if error/np.absolute(num_grad)>threshold:
                    print("Wrong grad at Wh[{},{}]".format(i,j))
                    print("Numerical grad:  ",num_grad)
                    print("Theoretical grad:",d_Wh[i,j])
        print("\n\n\n")
        # grad check for Wo
        for i in range(e_Wo.shape[0]):
            for j in range(e_Wo.shape[1]):
                self.Wo[i,j] += d_theta
                loss_plus = self.loss(X_seq, y_seq)
                self.Wo[i,j] += -2*d_theta
                loss_minus = self.loss(X_seq, y_seq)
                self.Wo[i,j] += d_theta

                num_grad = (loss_plus-loss_minus)/(2*d_theta)
                error = np.absolute(d_Wo[i,j]-num_grad)
                e_Wo[i,j] = error
                if(error/np.absolute(num_grad)>threshold):
                    print("Wrong grad at Wo[{},{}]".format(i,j))
                    print("Numerical grad:  ",num_grad)
                    print("Theoretical grad:",d_Wo[i,j])                    
        print("\n\n\n")
        # grad check for bo
        for i in range(e_bo.shape[0]):
            for j in range(e_bo.shape[1]):
                self.bo[i,j] += d_theta
                loss_plus = self.loss(X_seq, y_seq)
                self.bo[i,j] += -2*d_theta
                loss_minus = self.loss(X_seq, y_seq)
                self.bo[i,j] += d_theta

                num_grad = (loss_plus-loss_minus)/(2*d_theta)
                error = np.absolute(d_bo[i,j]-num_grad)
                e_bo[i,j] = error
                if(error/np.absolute(num_grad)>threshold):
                    print("Wrong grad at bo[{},{}]".format(i,j))
                    print("Numerical grad:  ",num_grad)
                    print("Theoretical grad:",d_bo[i,j])                    
        # grad check for bxh
        for i in range(self.bxh.shape[0]):
            for j in range(self.bxh.shape[1]):
                self.bxh[i,j] += d_theta
                loss_plus = self.loss(X_seq, y_seq)
                self.bxh[i,j] -= 2*d_theta
                loss_minus = self.loss(X_seq, y_seq)
                self.bxh[i,j] += d_theta
                num_grad = (loss_plus-loss_minus)/(2*d_theta)
                error = np.absolute(d_bxh[i,j]-num_grad)
                e_bxh[i,j] = error
                if error/np.absolute(num_grad)>threshold:
                    print("Wrong grad at bxh[{},{}]".format(i,j))
                    print("Numerical grad:  ",num_grad)
                    print("Theoretical grad:",d_bxh[i,j])
        print("\n\n\n")