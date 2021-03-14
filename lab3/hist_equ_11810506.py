from typing import io
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def hist_equ_11810506(image_input):
    """ Histogram equalization of a grayscale image. """
    input_img = io.imread(image_input)
    r, c = input_img.shape
    output_img = np.zeros([r, c], dtype=np.uint8)
    input_histogram = []
    output_histogram = []

    # histogram of input image
    # pdf
    for i in range(256):
        input_histogram.append(np.sum(input_img == i) / (r * c))

    # get cumulative distribution function
    cdf = []
    sum = 0
    for i in range(len(input_histogram)):
        sum = sum + input_histogram[i]
        cdf.append(sum)

    # cdf = 255 * cdf / cdf[-1]

    for i in range(r):
        for j in range(c):
            output_img[i, j] = ((256 - 1)) * cdf[input_img[i, j]]

    for i in range(256):
        output_histogram.append(np.sum(output_img == i) / (r * c))

    io.imsave(image_input.strip(".tif") + "_11810506.tif", output_img)

    n = np.arange(256)
    plt.plot(n, input_histogram)
    plt.savefig(image_input.strip(".tif") + "_input_hist_11810506.tif")
    plt.close()
    plt.plot(n, output_histogram)
    plt.savefig(image_input.strip(".tif") + "_output_hist_11810506.tif")
    plt.close()


    return (
        image_input + "_11810506.tif", image_input + "output_hist_11810506.tif",
        image_input + "input_hist_11810506.tif")


hist_equ_11810506("Q3_1_1.tif")
hist_equ_11810506("Q3_1_2.tif")
