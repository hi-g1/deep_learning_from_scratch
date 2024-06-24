import numpy as np
from gradient.numerical_diff import numerical_gradient
from functions.sigmoid_function import sigmoid_function
from functions.cross_entropy_error import CEE
from functions.softmax_function import softmax_function

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.param={}
        self.param['W1']=weight_init_std*\
            np.random.randn(input_size,hidden_size)
        self.param['b1']=np.zeros(hidden_size)
        self.param['W2']=weight_init_std*\
            np.random.randn(hidden_size,output_size)
        self.param['b2']=np.zeros(output_size)

    def predict(self,x):
        W1,W2=self.param['W1'], self.param['W2']
        b1,b2=self.param['b1'], self.param['b2']

        a1=np.dot(x,W1)+b1
        z1=sigmoid_function(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax_function(a2)

        return y
    
    def loss(self, x,t):
        y= self.predict(x)

        return CEE(y,t)
    
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)
        t=np.argmax(t, axis=1)

        accuracy= np.sum(y==t)/x.shape[0]
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_w=lambda W: self.loss(x,t)

        grads={}
        grads['W1']=numerical_gradient(loss_w, self.param['W1'])
        grads['b1']=numerical_gradient(loss_w, self.param['b1'])
        grads['W2']=numerical_gradient(loss_w, self.param['W2'])
        grads['b2']=numerical_gradient(loss_w, self.param['b2'])

        return grads
        
def main():
    net=TwoLayerNet(input_size=784, hidden_size=100,output_size=10)
    x=np.random.randn(100,784)
    t=np.random.randn(100,10)
    y=net.predict(x)

    grads=net.numerical_gradient(x,t)
    print(grads)

if __name__=="__main__":
    main()
        