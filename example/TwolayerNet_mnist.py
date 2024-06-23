import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
load_mnist( normalize=False, one_hot_label=True)

train_loss_list=[]

iters_num=10000
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1
network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for _ in range(iters_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    grad=network.numerical_gradient(x_batch,t_batch)

    for key in list(network.param.keys()):
        network.param[key] -= learning_rate*grad[key]
    
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    print(f"{loss=}")