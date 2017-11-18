#Decoding script
#####################
import cv2
import numpy as np
from matplotlib import pyplot as plt

g=0.1

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


#####################
#Read the original image, the watermarked image and the watermark
#####################
original = cv2.imread('lena512.bmp',0)
D = cv2.imread("WatermarkedLena.bmp",0)
Watermark = np.loadtxt("Watermark", delimiter=' ')
K=Watermark.__len__()
print K
rows,cols = D.shape
#####################
#DCT of original image
#####################
Origif = np.float32(original)/255.0
Origidct = cv2.dct(Origif)

#####################
#DCT of watermarked image
#####################
Df = np.float32(D)/255.0
Ddct = cv2.dct(Df)

#####################
#Extracting high AC
#####################
Ddctp=Ddct.copy()
Ddctp[0][0]=0
indices =largest_indices(Ddctp, K)

#####################
#Extracting watermark from WI
#####################
ExtractedWatermark = (Ddct[indices]-Origidct[indices])/(g*Origidct[indices])
print ExtractedWatermark


#####################
#Evaluate the watermark
#####################
Similarity = (np.matrix(ExtractedWatermark)*np.matrix(Watermark).T)/(np.sqrt(np.matrix(ExtractedWatermark)*np.matrix(ExtractedWatermark).T))


#####################
#Tests
#####################
print Similarity
plt.subplot(121),plt.imshow(original , cmap = 'gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(D , cmap = 'gray')
plt.title('Watermarked image'), plt.xticks([]), plt.yticks([])

plt.show()
