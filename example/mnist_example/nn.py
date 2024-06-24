import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.abspath(__file__),'..','..'))
from functions.sigmoid_function import sigmoid_function
from functions.identity_function import identity_function
from typing import Dict


def init_network() -> Dict:
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) # 2x3
    network['b1']=np.array([0.1,0.2,0.3]) # 1x3
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]]) # 3x2
    network['b2']=np.array([0.1,0.2]) # 1x2
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]]) # 2x2
    network['b3']=np.array([0.1,0.2]) # 1x2

    return network

def forward(network: Dict, x: np.ndarray) -> np.ndarray:
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid_function(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid_function(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)

    return y

network = init_network()
x=np.random.rand(1,2)
print(f"{forward(network,x)=}")





