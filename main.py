import pywt
import matplotlib.pyplot as plt
import numpy as np
import cv2
from timeit import default_timer as timer


# noinspection PyPep8Naming,DuplicatedCode
def WaveletDenoising():
    imagePath = 'C:\\Users\\Artyom\\Downloads\\FromRawFLX.jpeg'
    outputImagePath = 'C:\\Users\\Artyom\\Downloads\\WaveletDenoised.jpeg'
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    (l, a, b) = cv2.split(img)
    waveletName = 'db1'
    colorNoise = 100
    lightnessNoise = 100

    # Lightness
    lArray = np.asarray(l)
    (data, (coefficients1, coefficients2, coefficients3)) = pywt.dwt2(lArray, waveletName)
    coefficients1 = pywt.threshold(coefficients1, lightnessNoise)
    coefficients2 = pywt.threshold(coefficients2, lightnessNoise)
    coefficients3 = pywt.threshold(coefficients3, lightnessNoise)
    lArray = pywt.idwt2((data, (coefficients1, coefficients2, coefficients3)), waveletName)

    start = timer()

    end = timer()
    print('Wavelet denoising for l channel is : ', (end - start) * 1000)


    # A
    aArray = np.asarray(a)
    (data, (coefficients1, coefficients2, coefficients3)) = pywt.dwt2(aArray, waveletName)
    coefficients1 = pywt.threshold(coefficients1, colorNoise)
    coefficients2 = pywt.threshold(coefficients2, colorNoise)
    coefficients3 = pywt.threshold(coefficients3, colorNoise)

    (data2, (coefficients12, coefficients22, coefficients32)) = pywt.dwt2(data, waveletName)
    coefficients12 = pywt.threshold(coefficients12, colorNoise)
    coefficients22 = pywt.threshold(coefficients22, colorNoise)
    coefficients32 = pywt.threshold(coefficients32, colorNoise)
    data = pywt.idwt2((data2, (coefficients12, coefficients22, coefficients32)), waveletName)
    data = np.delete(data, np.s_[-1:], axis=0)
    data = np.delete(data, np.s_[-1:], axis=1)
    aArray = pywt.idwt2((data, (coefficients1, coefficients2, coefficients3)), waveletName)

    # B
    bArray = np.asarray(b)
    (data, (coefficients1, coefficients2, coefficients3)) = pywt.dwt2(bArray, waveletName)
    coefficients1 = pywt.threshold(coefficients1, colorNoise)
    coefficients2 = pywt.threshold(coefficients2, colorNoise)
    coefficients3 = pywt.threshold(coefficients3, colorNoise)
    (data2, (coefficients12, coefficients22, coefficients32)) = pywt.dwt2(data, waveletName)
    coefficients12 = pywt.threshold(coefficients12, colorNoise)
    coefficients22 = pywt.threshold(coefficients22, colorNoise)
    coefficients32 = pywt.threshold(coefficients32, colorNoise)
    data = pywt.idwt2((data2, (coefficients12, coefficients22, coefficients32)), waveletName)
    data = np.delete(data, np.s_[-1:], axis=0)
    data = np.delete(data, np.s_[-1:], axis=1)
    bArray = pywt.idwt2((data, (coefficients1, coefficients2, coefficients3)), waveletName)

    lArray = np.delete(lArray, np.s_[-1:], axis=0)
    aArray = np.delete(aArray, np.s_[-1:], axis=0)
    bArray = np.delete(bArray, np.s_[-1:], axis=0)
    print(img.shape)
    print(lArray.shape)
    lArray = lArray.astype(np.uint8)
    aArray = aArray.astype(np.uint8)
    bArray = bArray.astype(np.uint8)
    img = cv2.merge([lArray, aArray, bArray])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    cv2.imwrite(outputImagePath, img)




if __name__ == '__main__':
    WaveletDenoising()
