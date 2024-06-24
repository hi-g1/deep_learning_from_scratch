import sys,os
sys.path.append(os.path.join(os.path.abspath(__file__),'..','..'))
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np
import time

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()
    time.sleep(1) # 이미지뷰터 로딩 시간 대기

(x_train, t_train), (x_test, t_test) = \
load_mnist(flatten=True, normalize=False)

img=x_train[0]
label=t_train[0]

# print(img.shape) # 플랫튼이라 1x28x28=784 나옴
img=img.reshape(28,28)
img_show(img)




