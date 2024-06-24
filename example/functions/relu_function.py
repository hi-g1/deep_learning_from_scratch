import numpy as np
import matplotlib.pyplot as plt

def relu_function(x: np.ndarray) -> np.ndarray:
    return np.maximum(0,x)

def main() -> None:
    x = np.arange(-5,5,0.1)
    y = relu_function(x)
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    main()