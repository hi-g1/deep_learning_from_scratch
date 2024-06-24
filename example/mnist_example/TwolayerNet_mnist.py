import numpy as np
import os,sys
sys.path.append(os.path.join(os.path.abspath(__file__),'..','..'))
from dataset.mnist import load_mnist
from model.TwoLayerNet import TwoLayerNet
import yaml

with open('example\mnist_example\config.yaml', 'r') as f:
    config=yaml.safe_load(f)

(x_train, t_train), (x_test, t_test) = \
load_mnist( normalize=False, one_hot_label=True)

train_loss_list=[]

iters_num=config['iters_num']
batch_size=config['batch_size']
learning_rate = config['learning_rate']
input_size = config['input_size']
hidden_size = config['hidden_size']
output_size = config['output_size']
train_size=x_train.shape[0]

network=TwoLayerNet(input_size,hidden_size,output_size)

# 랜덤 시드 설정
np.random.seed(0)

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