#Encoding script
#####################
import cv2
import numpy as np
from matplotlib import pyplot as plt

g=0.1#Higher to 1 makes it visible
K=64

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))



#####################
#Reading image
#####################
img = cv2.imread('lena512.bmp',0) # 1 chan, grayscale!
height = np.size(img, 0)
width = np.size(img, 1)
#Host image B to which we will apply the watermark
B=img.copy()
rows,cols = B.shape


#####################
#Generating watermark
#####################
h1 = np.random.randn(1,K)
plusminus1=np.sign(h1)
for i in range(0,K):
    if plusminus1[0][i] < 0 : plusminus1[0][i]=0 #message of K elements to hide


#####################
#DCT of original image
#####################
Bf = np.float32(B)/255.0
Bdct = cv2.dct(Bf)
Btest=Bdct.copy()

#####################
#Detecting the K max alternative coef of the DCT
#####################
Bdctp=Bdct.copy()
Bdctp[0][0]=0
indices =largest_indices(Bdctp, K)


#####################
#Watermarking
#####################
Bdct[indices]=Bdct[indices]+g*plusminus1*Bdct[indices]


#####################
#Storing watermarked image and the watermark
#####################
B = cv2.idct(Bdct)
final=np.float32(B)*255.0
aff1=Bdct-Btest
aff=(cv2.idct(aff1))
watermark=np.uint8(np.float32(aff)*255.0)
np.savetxt('Watermark', plusminus1, fmt='%i')
cv2.imwrite('WatermarkedLena.jpg',final)

#####################
#Compute the PSNR
#####################
err=psnr(final,img)

#####################
#Tests
#####################
print err
plt.subplot(131),plt.imshow(img , cmap = 'gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(final , cmap = 'gray')
plt.title('Watermarked image'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow( watermark, cmap = 'gray')
plt.title('Watermark'), plt.xticks([]), plt.yticks([])

plt.show()
