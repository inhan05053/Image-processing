import numpy as np
import cv2
if __name__ == '__main__':
    msize=50
    msizeDivTwo=msize//2
    y,x = np.mgrid[-(msizeDivTwo):msizeDivTwo+1,-(msizeDivTwo):msizeDivTwo+1 ]
    x=x*220
    dst = (((x * x) + (x * x))) ** 0.5


    print(x)
    cv2.imshow('dst', x/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print(dst*2)
    print(256*256)
    print(256*256)
    print(256*256)


