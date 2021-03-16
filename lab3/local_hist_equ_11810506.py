from typing import io
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time


def part_img(input_img, m, n, m_size):
    sums = 0
    step = int((m_size - 1) / 2)
    for i in range(m - step, m + step + 1):
        for j in range(n - step, n + step + 1):
            if 0 <= i < input_img.shape[0] and 0 <= j < input_img.shape[0]:
                if input_img[i, j] <= input_img[m, n]:
                    sums = sums + 1

    return (255) * sums / (m_size * m_size)


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
    for i in range(0, r):
        for j in range(0, c):
            output_img[i, j] = part_img(input_img, i, j, m_size)
            # print(i, j)
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


time_start = time.time()
local_hist_equ_11810506("Q3_3.tif", 11)
time_end = time.time()
string = "totally cost %.2f s" % (time_end - time_start)
print(string)
