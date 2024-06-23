import numpy as np
from numerical_diff import numerical_diff,function_2

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x=init_x

    for _ in range(step_num):
        grad=numerical_diff(f,x)
        pre_x = x
        x-=x-lr*grad
        #print(f"{pre_x=} grad {x=}")
    
    return x

def main():
    init_x=np.array([-3.,4.])
    x=gradient_descent(function_2,init_x=init_x,lr=0.1)
    print(x)
    

if __name__=="__main__":
    main()