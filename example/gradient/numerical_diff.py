import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..'))
from tqdm import tqdm

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return np.sum(x**2)

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x 형태처럼 배열 생성

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    total_steps = x.size

    with tqdm(total=total_steps, desc="Calculating gradient", unit="step") as pbar:
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)

            grad[idx] = (fxh1 - fxh2) / (2 * h)  # 중심차분
            x[idx] = tmp_val  # 값 복원
            it.iternext()
            pbar.update(1)

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


