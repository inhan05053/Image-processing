import cv2
import numpy as np

def my_bilinear(src, scale):
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)
    dst = np.zeros((h_dst, w_dst))
    # interbilinear polation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            # 참고로 꼭 한줄로 구현해야 하는건 아닙니다 여러줄로 하셔도 상관없습니다.(저도 엄청길게 구현했습니다.)
            t,s=float(col/scale),float(row/scale)
            m,n= int(t), int(s)
            if m>= h-1 or n>= w-1 :
                dst[col,row]=src[m,n]
                continue
            dst[col,row] =src[m,n]*(m+1-t)*(n+1-s)+src[m+1,n]*(n+1-s)*(t-m)+src[m,n+1]*(s-n)*(m+1-t)+src[m+1,n+1]*(t-m)*(s-n)
    return dst

if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/2
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)
    cv2.waitKey()
    cv2.destroyAllWindows()