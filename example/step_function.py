import numpy as np
import matplotlib.pyplot as plt

def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=np.int16)

def main() -> None:
    x = np.arange(-5., 5., 0.1)
    y = step_function(x)
    plt.ylim(-0.1, 1.1)
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()