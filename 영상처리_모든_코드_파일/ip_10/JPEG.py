import numpy as np
import cv2
import time

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def img2block(src, n=8):
    blocks= []
    h,w = src.shape
    for i in range(h//8):
        for j in range(w//8):
            blocks.append(src[n*i:(n*i+8),j*n:(j*n+8)])
    return np.array(blocks).astype(np.int32)


def DCT(block, n=8):

    dst = np.zeros((n,n))
    y, x = np.mgrid[0:n, 0:n]
    for v_ in range(n):
        for u_ in range(n):
            tmp = block * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            dst[v_,u_] = np.sum(tmp) * C(v_,n=n) * C(u_,n=n)
    return np.round(dst)

def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5


def my_zigzag_scanning(block, mode='encoding', block_size=8):


    if mode=='decoding':
        num=0
        zeros=np.zeros((block_size,block_size))
        b=[0]*64
        b[0:len(block)]=block[0:len(block)]
        for k in range(8):
            i=0
            j=k
            if k%2==0:
                j=0
                i=k
                for n in range(k+1):
                    if b[num] == 'EOB':
                        return zeros
                    zeros[i,j]= b[num]
                    num+=1
                    i-=1
                    j+=1
            else :
                for n in range(k+1):
                    if b[num] == 'EOB':
                        return zeros
                    zeros[i,j]= b[num]
                    num+=1
                    i+=1
                    j-=1
        for k in range(7):
            i=7
            j=k+1
            if (k+1)%2==0:
                i=k+1
                j=7
                for n in range(7-k):
                    if b[num] == 'EOB':
                        return zeros
                    zeros[i,j]= b[num]
                    num+=1
                    i+=1
                    j-=1
            else :
                for n in range(7-k):
                    if b[num] == 'EOB':
                        return zeros
                    zeros[i,j]= b[num]
                    num+=1
                    i-=1
                    j+=1
        return zeros

    else :
        z=[]
        for k in range(8):
            i=0
            j=k
            if k%2==0:
                j=0
                i=k
                for n in range(k+1):
                    z.append(block[i,j])
                    i-=1
                    j+=1
            else :
                for n in range(k+1):
                    z.append(block[i,j])
                    i+=1
                    j-=1
        for k in range(7):
            i=7
            j=k+1
            if (k+1)%2==0:
                i=k+1
                j=7
                for n in range(7-k):
                    z.append(block[i,j])
                    i+=1
                    j-=1
            else :
                for n in range(7-k):
                    z.append(block[i,j])
                    i-=1
                    j+=1
        k=63
        while (z[k]==0):
            z.pop()
            k-=1
            if k==0:
                break
    return z

def DCT_inv(block, n = 8):

    dst = np.zeros((n,n))
    u_, v_ = np.mgrid[0:n, 0:n]

    for i in range(n):
        for j in range(n):
            block[i,j]  = block[i,j] * C(i,n=n) * C(j,n=n)
    for y in range(n):
        for x in range(n):
            tmp = block *np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            dst[x,y] = np.sum(tmp)
    return np.round(dst)

def block2img(blocks, src_shape, n = 8):

    h,w = src_shape
    dst= np.zeros((h,w))

    k=0
    for i in range(h//8):
        for j in range(w//8):
            dst[n*i:(n*i+8),j*n:(j*n+8)]= blocks[k]
            k+=1
    return dst.astype(np.int32)

def Encoding(src, n=8):

    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)

    #subtract 128
    blocks =  blocks - 128
    #DCT

    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)


    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)


    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))
    return zz, src.shape

def Decoding(zigzag, src_shape, n=8):

    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)
    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)
    # add 128

    blocks_idct += 128
    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)
    dst= ((dst-np.min(dst))/ np.max(dst-np.min(dst))*255).astype(np.uint8)
    return dst



def main():
    start = time.time()
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    comp, src_shape = Encoding(src, n=8)

    # 과제의 comp.npy, src_shape.npy를 복구할 때 아래 코드 사용하기(위의 2줄은 주석처리하고, 아래 2줄은 주석 풀기)
    # comp = np.load('comp.npy', allow_pickle=True)
    # src_shape = np.load('src_shape.npy')


    recover_img = Decoding(comp, src_shape, n=8).astype(np.uint8)
    total_time = time.time() - start

    print('time : ', total_time)
    if total_time > 45:
        print('감점 예정입니다.')
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
