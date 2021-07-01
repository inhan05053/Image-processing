import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):

    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w), dtype=np.uint8)
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')


        pad_img[:p_h, p_w:p_w+w]= pad_img[p_h, p_w:p_w+w]

        pad_img[p_h+h:, p_w:p_w+w]= pad_img[h+p_h-1, p_w:p_w+w]

        pad_img[:, :p_w]= pad_img[:, p_w:p_w+1]

        pad_img[:, p_w+w:]= pad_img[:, w+p_w-1:w+p_w]

    else:
        print('zero padding')

    return pad_img

def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)

    if ftype == 'average':
        print('average filtering')

        #np.full 함수를 이용해 numpy를 만듬 average이므로 같은 값
        mask = np.full((fshape[0], fshape[1]), 1/(fshape[0]*fshape[1]))



    elif ftype == 'sharpening':
        print('sharpening filtering')
        # -1/(가로*세로) 값을 전제에 넣어주고 중앙 한값에 2- 1/(가로*세로)을 넣는다.
        mask = np.full((fshape[0], fshape[1]), -1 / (fshape[0] * fshape[1]))
        mask[fshape[0]//2][fshape[1]//2] = 2 -1 / (fshape[0] * fshape[1])






    #sum을 이용하여 filter된 값을 구해 저장
    for c in range(h):
        for r in range(w):
            image_piece = src_pad[c:c+fshape[0],r:r+fshape[1]]
            val= np.sum(mask*image_piece)
            val = np.clip(val, 0, 255)
            src_pad[c + (fshape[0] // 2)][r + (fshape[1] // 2)] = val
            # if np.sum(mask*image_piece)>=255:
            #     src_pad[c+(fshape[0]//2)][r+(fshape[1]//2)]=255
            # elif np.sum(mask*image_piece)<0:
            #     src_pad[c+(fshape[0]//2)][r+(fshape[1]//2)]=0
            # else:
            #     src_pad[c+(fshape[0]//2)][r+(fshape[1]//2)]=(np.sum(mask*image_piece))


    src_pad = (src_pad+0.5).astype(np.uint8)
    return src_pad[fshape[0]//2:h+fshape[0]//2,fshape[1]//2:w+fshape[1]//2]


if __name__ == '__main__':
    src = cv2.imread('../../ImageProcessing/Lena.png', cv2.IMREAD_GRAYSCALE)
    # repetition padding test
    rep_test = my_padding(src, (20,20), 'repetition')

    # 3x3 filter
    # dst_average = my_filtering(src, 'average', (3,3))
    # dst_sharpening = my_filtering(src, 'sharpening', (3,3))

    #원하는 크기로 설정s
    dst_average = my_filtering(src, 'average', (5,7))
    dst_sharpening = my_filtering(src, 'sharpening', (5,7))

    # 11x13 filter
    # dst_average = my_filtering(src, 'average', (11,13), 'repetition')
    # dst_sharpening = my_filtering(src, 'sharpening', (11,13), 'repetition')

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
