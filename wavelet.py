import numpy as np
import os
import pywt
import pywt.data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


m = mpimg.imread('static/images.jpg')
c = pywt.wavedec2(m, 'db5', mode='periodization', level=2)
imgr = pywt.waverec2(c, 'db5', mode='periodization')
imgr = np.uint8(imgr)
print("****************", imgr)

cA2 = c[0]
(cH1, cV1, cD1) = c[-1]
(cH2, cV2, cD2) = c[-2]
print("****************")
plt.figure(figsize=(20, 20))
print("****************")

plt.subplot(2, 2, 1)
plt.imshow(cA2, cmap=plt.cm.gray)
plt.title('cA2: Approximation Coeff.', fontsize=30)
print("****************")

plt.subplot(2, 2, 2)
plt.imshow(cH2, cmap=plt.cm.gray)
plt.title('cA2: Horizontal Detailed Coeff.', fontsize=30)
print("****************")

plt.subplot(2, 2, 3)
plt.imshow(cV2, cmap=plt.cm.gray)
plt.title('cV2: Vertical Detailed Coeff.', fontsize=30)

plt.subplot(2, 2, 4)
plt.imshow(cD2, cmap=plt.cm.gray)
plt.title('cD2: Diagonal Detailed Coeff.', fontsize=30)

# arr, coeff_slices = pywt.coeffs_to_array(c)
# plt.figure(figsize=(20, 20))
# plt.imshow(arr, cmap=plt.cm.gray)
# plt.title('All Wavelet Coeffs. upto level2', fontsize=30)

plt.figure()
plt.imshow(imgr, cmap=plt.cm.gray)
plt.title('Reconstructed Image', fontsize=10)
plt.show()