import skimage as io
from typing import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def laplace_sharpen(input_image, c, choice):
    input_image_cp = io.imread(input_image)
    m, n = input_image_cp.shape
    input_image_cp = np.pad(input_image_cp, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    process_image = np.zeros(input_image_cp.shape)
    scaled_image = np.copy(input_image_cp)
    output_image = np.copy(input_image_cp)

    laplace_filter1 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])
    laplace_filter2 = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ])

    if choice == 1:
        laplace_filter = laplace_filter1
        name = input_image.strip(".tif") + " laplace_sharpen_type 1_11810506.tif"
    else:
        laplace_filter = laplace_filter2
        name = input_image.strip(".tif") + " laplace_sharpen_type 2_11810506.tif"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            process_image[i, j] = np.sum(laplace_filter * input_image_cp[i - 1:i + 2, j - 1:j + 2])

    a = np.min(process_image)
    b = 255 / np.max(process_image)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            scaled_image[i, j] = int((process_image[i, j] - a) * b)
            output_image[i, j] = input_image_cp[i, j] + c * process_image[i, j]
            if process_image[i, j] < 0:
                process_image[i, j] = 0

    process_image = process_image[1:m - 1, 1:n - 1]
    scaled_image = scaled_image[1:m - 1, 1:n - 1]
    output_image = output_image[1:m - 1, 1:n - 1]
    plt.figure(figsize=(6, 6))

    plt.subplot(221)
    plt.title('input_image', fontsize=10)
    plt.imshow(input_image_cp, cmap='gray')

    plt.subplot(222)
    plt.title('laplace_image', fontsize=10)
    plt.imshow(process_image, cmap='gray')

    plt.subplot(223)
    plt.title('scaled_image', fontsize=10)
    plt.imshow(scaled_image, cmap='gray')

    plt.subplot(224)
    plt.title('output_image', fontsize=10)
    plt.imshow(output_image, cmap='gray')
    plt.savefig(name)
    plt.close()

    return 0



