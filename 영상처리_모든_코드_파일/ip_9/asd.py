import cv2
import numpy as np
import matplotlib.pyplot as p_
import matplotlib.cm as cm
def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)


def my_DCT(src, n=8):
    ###############################
    # TODO                        #
    # my_DCT 완성                 #
    # src : input image           #
    # n : block size              #
    ###############################
    (h, w) = src.shape
    dct_img = (src.copy()).astype(np.float)
    # 이미지를 복사해와서 실수형으로 변경
    dst = np.zeros((h, w))
    # 새로운 이미지 정의
    mask = np.zeros((n, n), dtype=np.float)
    # 마스크를 정의한다 n*n 형태로 구현한다.
    alpha_1 = 0
    # C(u)값 변수
    alpha_2 = 0
    # C(v)값 변수
    for u in range(h // n):
        for v in range(w // n):
            if u == 0:
                alpha_1 = np.sqrt(1 / n)
            elif u != 0:
                alpha_1 = np.sqrt(2 / n)
            if v == 0:
                alpha_2 = np.sqrt(1 / n)
            elif v != 0:
                alpha_2 = np.sqrt(2 / n)

            for mask_row in range(n):
                for mask_col in range(n):
                    for row in range(n):
                        for col in range(n):
                            mask[row][col] = np.cos((2 * row + 1) * np.pi * mask_row / (2 * n)) * np.cos(
                                (2 * col + 1) * np.pi * mask_col / (2 * n))
                    dst[u * n + mask_col][v * n + mask_row] = alpha_1 * alpha_2 * np.sum(
                        dct_img[u * n: (u + 1) * n, v * n: (v + 1) * n] * mask)

    return dst

if __name__ == '__main__':
    block_size = 4
    src = np.ones((block_size, block_size))
    dst = my_DCT(src, n=block_size)

    dst = my_normalize(dst)
    cv2.imshow('my DCT', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


