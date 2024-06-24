import numpy as np
import os,sys
sys.path.append(os.path.join(os.path.abspath(__file__),'..','..'))
from dataset.mnist import load_mnist
import pickle
from typing import Dict
from functions.sigmoid_function import sigmoid_function
from functions.softmax_function import softmax_function

def get_data() -> None:
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)

    return x_test, t_test
    
def init_network():
    with open("example\mnist_example\sample_weight.pkl", 'rb') as f:
        network=pickle.load(f)

    return network

def predict(network: Dict, x: np.ndarray) -> np.ndarray:
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid_function(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid_function(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax_function(a3)

    return y

x,t = get_data()
network=init_network()

batch_size = 100
accuracy_cnt=0
for idx in range(0,len(x),batch_size):
    x_batch=x[idx:idx+batch_size]
    y_batch=predict(network, x_batch)
    p=np.argmax(y_batch, axis=1) # 행 기준으로 argmax 적용
    accuracy_cnt += np.sum( p==t[idx:idx+batch_size], dtype=np.int16)

print(f"정확도 = {accuracy_cnt/len(x)}")

