import numpy as np

# 오버플로 문제 발생
"""
def softmax_function(x: np.ndarray) -> np.ndarray:
    return np.exp(x)/np.sum(np.exp(x))
"""

def softmax_function(x: np.ndarray) -> np.ndarray:
    c=np.max(x)
    x_exp=np.exp(x-c) # 오버플로 방지용
    x_sum=np.sum(x_exp)
    y=x_exp/x_sum

    return y