import cv2
import numpy as np

def dilation(B, S):
    h,w = len(B),len(B[0])
    h_s,w_s =  len(S),len(S[0])
    dst= np.zeros((h+2*(h_s//2),w+2*(w_s//2)), dtype=np.uint8)  #더 큰 공간을 만들어 B의 index 오류를 맞아준다

    for row in range(h):
        for col in range(w):
            if B[row][col]==1:
                dst[row:row+h_s,col:col+w_s] = S[:,:]  # 중앙이 1이라면 주변을 모두 1로 바꿔준다.

    dst= dst[h_s//2:h+h_s//2,w_s//2:w+w_s//2]  #B의 크기로 크기 조절을 해준다.
    return dst

def erosion(B, S):
    h,w = len(B),len(B[0])
    h_s,w_s =  len(S),len(S[0])
    dst= np.zeros((h+2*(h_s//2),w+2*(w_s//2)), dtype=np.uint8)
    B_plus= np.zeros((h+2*(h_s//2),w+2*(w_s//2)), dtype=np.uint8)

    B_plus[h_s // 2:h + h_s // 2, w_s // 2:w + w_s // 2] = B[:,:]
    for row in range(h):
        for col in range(w):
            if  B_plus[row:row+h_s,col:col+w_s].all() == S.all() :  #b와 s의 값이 모두 1이라면 중앙 값을 1로 설정한다.
                dst[row+h_s//2,col+w_s//2]=1
    dst = dst[h_s // 2:h + h_s // 2, w_s // 2:w + w_s // 2]  #B의 크기로 다시 조절해준다.
    return dst

def opening(B, S):  #erosion 과 dilation을 차례로 수행

    B=erosion(B,S)
    dst=dilation(B,S)
    return dst

def closing(B, S):  #dilation 과 erosion을 차례로 수행

    B=dilation(B,S)
    dst=erosion(B,S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


