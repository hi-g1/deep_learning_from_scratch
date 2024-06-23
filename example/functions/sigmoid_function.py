import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    return expit(x)

def main() -> None:
    x = np.arange(-5,5,0.1)
    y = sigmoid_function(x)
    plt.plot(x,y)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()