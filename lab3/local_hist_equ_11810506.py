from typing import io
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def hist_equ(part):
    """ Histogram equalization of a grayscale image. """
    input_img = part
    r, c = input_img.shape
    x = int((r - 1) / 2)
    sum = 0
    for i in range(int(input_img[x, x]+1)):
        sum = sum +np.sum(input_img == i) / (r * c)
    # get cumulative distribution function

    return (255) * sum

def part_img(input_img, m, n, m_size):
    step = int((m_size - 1) / 2)
    part = np.zeros([m_size, m_size], dtype=np.uint8)

    for i in range(m - step, m + step):
        for j in range(n - step, n + step):
            if i >= 0 and i < input_img.shape[0] and j >= 0 and j < input_img.shape[0]:
                part[i - (m - step), j - (n - step)] = input_img[i, j]
    return part

def local_hist_equ_11810506(input_image, m_size):
    input_img = io.imread(input_image)
    output_img = np.zeros(input_img.shape, dtype=np.uint8)
    r, c = input_img.shape
    input_histogram = []  # Distribution of input pixels
    output_histogram = []  # Distribution of output pixels

    # Count input
    for i in range(256):
        input_histogram.append(np.sum(input_img == i))

    # local histogram equalization
    for i in range(0,r):
        for j in range(0,c):
            output_img[i, j] = hist_equ(part_img(input_img, i, j, m_size))
            print(i,j)
    # Count output

    for i in range(256):
        output_histogram.append(np.sum(output_img == i))

    io.imsave("Q3_3" + "_11810506.tif", output_img)

    n = np.arange(256)
    plt.plot(n, input_histogram)
    plt.savefig("Q3_3" + "_input_hist_11810506.tif")
    plt.close()
    plt.plot(n, output_histogram)
    plt.savefig("Q3_3" + "_output_hist_11810506.tif")
    plt.close()
    return


local_hist_equ_11810506("Q3_3.tif", 21)