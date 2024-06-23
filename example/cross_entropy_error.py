import numpy as np

def CEE(y: np.ndarray, t:np.ndarray) -> np.ndarray:
    """
    t가 원-핫 인코딩 시
    """
    delta= 1e-7 # -inf 방지

    if y.ndim == 1: # 1차원 -> 2차원 배열
        y = y.reshape(1,y.size) # size(모든 원소 수)
        t = t.reshape(1,t.size)

    batch_size = y.shape[0] # 배치크기
    return -np.sum(t*np.log(y + delta))/batch_size

def main() -> None:
    x = np.random.random_integers(0,9,10,)
    mask = np.array( x==2, dtype=np.int8)
    print(x)
    print(mask)
    print(CEE(x,mask))

if __name__ == "__main__":
    main()