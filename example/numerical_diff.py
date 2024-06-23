import numpy as np
import matplotlib.pyplot as plt

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return np.sum(x**2)

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x) # x형태처러 배열 생성

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x[idx]) # f(x+h)

        x[idx]=tmp_val-h
        fxh2=f(x[idx]) # f(x-h)

        grad[idx]=(fxh1-fxh2)/(2*h) # 중심차분
        x[idx]=tmp_val

    return grad


def main():
    print(numerical_gradient(function_2,np.array([3.0,100.0])))
    
    """
    x=np.arange(0.0,20.0,0.1)
    y=function_1(x)

    print(numerical_diff(function_1,x))

    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()
    plt.show()
    """

if __name__ == "__main__":
    main()


