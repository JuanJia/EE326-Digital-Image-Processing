import numpy as np
import skimage
from matplotlib import pyplot as plt, pyplot
import pylab
from matplotlib.pyplot import gray
import cv2
from skimage import io


def gradient_sharpen_robert_filter_abs(input_image):
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    r = [[-1, -1], [1, 1]]
    for i in range(h):
        for j in range(w):
            if (j + 2 < w) and (i + 2 <= h):
                process_img = img[i:i + 2, j:j + 2]
                list_robert = r * process_img
                img[i, j] = abs(list_robert[1, 1] + list_robert[0, 0]) + abs(list_robert[1, 0] + list_robert[0, 1])
    io.imsave(input_image.strip(".tif") + " gradient_sharpen_robert_filter_abs_11810506.tif", img)
    return 0


def gradient_sharpen_robert_filter_sqrt(input_image):
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    r = [[-1, -1], [1, 1]]
    for i in range(h):
        for j in range(w):
            if (j + 2 < w) and (i + 2 <= h):
                process_img = img[i:i + 2, j:j + 2]
                list_robert = r * process_img
                img[i, j] = np.sqrt(np.square(list_robert[1, 1] + list_robert[0, 0]) + np.square(list_robert[1, 0] + list_robert[0, 1]))
    io.imsave(input_image.strip(".tif") + " gradient_sharpen_robert_filter_sqrt_11810506.tif", img)
    return 0


def gradient_sharpen_sobel_filter_abs(input_image):
    img = io.imread(input_image)
    h = img.shape[0]
    w = img.shape[1]
    image_new = np.zeros(img.shape, np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            sx = (img[i + 1][j - 1] + 2 * img[i + 1][j] + img[i + 1][j + 1]) - \
                 (img[i - 1][j - 1] + 2 * img[i - 1][j] + img[i - 1][j + 1])
            sy = (img[i - 1][j + 1] + 2 * img[i][j + 1] + img[i + 1][j + 1]) - \
                 (img[i - 1][j - 1] + 2 * img[i][j - 1] + img[i + 1][j - 1])
            image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))
    io.imsave(input_image.strip(".tif") + " gradient_sharpen_sobel_filter_abs_11810506.tif", image_new)
    return 0


def gradient_sharpen_sobel_filter_sqrt(input_image):
    img = io.imread(input_image)
    h = img.shape[0]
    w = img.shape[1]
    image_new = np.zeros(img.shape, np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            sx = (img[i + 1][j - 1] + 2 * img[i + 1][j] + img[i + 1][j + 1]) - \
                 (img[i - 1][j - 1] + 2 * img[i - 1][j] + img[i - 1][j + 1])
            sy = (img[i - 1][j + 1] + 2 * img[i][j + 1] + img[i + 1][j + 1]) - \
                 (img[i - 1][j - 1] + 2 * img[i][j - 1] + img[i + 1][j - 1])
            image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))
    io.imsave(input_image.strip(".tif") + " gradient_sharpen_sobel_filter_sqrt_11810506.tif", image_new)
    return 0

'''
def gradient_sharpen_prewitt_filter_abs(input_image):
    img = io.imread(input_image)
    h = img.shape[0]
    w = img.shape[1]
    image_new = np.zeros(img.shape, np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            sx = (img[i - 1][j - 1] + img[i - 1][j] + img[i - 1][j + 1]) - \
                 (img[i + 1][j - 1] + img[i + 1][j] + img[i + 1][j + 1])
            sy = (img[i - 1][j - 1] + img[i][j - 1] + img[i + 1][j - 1]) - \
                 (img[i - 1][j + 1] + img[i][j + 1] + img[i + 1][j + 1])
            image_new[i][j] = (np.abs(sx) + np.abs(sy))
    io.imsave(input_image.strip(".tif") + " gradient_sharpen_prewitt_filter_abs_11810506.tif", image_new)
    return image_new


def gradient_sharpen_prewitt_filter_sqrt(input_image):
    img = io.imread(input_image)
    h = img.shape[0]
    w = img.shape[1]
    image_new = np.zeros(img.shape, np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            sx = (img[i - 1][j - 1] + img[i - 1][j] + img[i - 1][j + 1]) - \
                 (img[i + 1][j - 1] + img[i + 1][j] + img[i + 1][j + 1])
            sy = (img[i - 1][j - 1] + img[i][j - 1] + img[i + 1][j - 1]) - \
                 (img[i - 1][j + 1] + img[i][j + 1] + img[i + 1][j + 1])
            image_new[i][j] = np.sqrt(np.square(sx) + np.square(sy))
    io.imsave(input_image.strip(".tif") + " gradient_sharpen_prewitt_filter_sqrt_11810506.tif", image_new)
    return image_new
'''

gradient_sharpen_sobel_filter_sqrt("Q4_1.tif")
