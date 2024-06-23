import numpy as np

def SSE(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return 0.5*np.sum((y-t)**2)

def main() -> None:
    x = np.random.random_integers(0,9,10,)
    mask = np.array( x==2, dtype=np.int8)
    print(x)
    print(mask)
    print(SSE(x,mask))

if __name__ == "__main__":
    main()