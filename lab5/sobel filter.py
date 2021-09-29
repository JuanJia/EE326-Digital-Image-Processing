import matplotlib.pyplot as plt
import numpy as np
import cv2


def sobel_filter_spatial(input_image):
    img = cv2.imread(input_image, 0)
    h = img.shape[0]
    w = img.shape[1]
    img_cp = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    image_new = np.zeros(img.shape)
    mask = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    for i in range(1, h):
        for j in range(1, w):
            image_new[i, j] = np.sum(mask * img_cp[i - 1:i + 2, j - 1:j + 2])

    a = np.min(image_new)
    b = 255 / (np.max(image_new) - a)
    for i in range(0, h):
        for j in range(0, w):
            image_new[i, j] = int((image_new[i, j] - a) * b)
    plt.imshow(image_new, cmap='gray')
    plt.axis('off')
    plt.savefig(input_image.strip(".tif") + " sobel_filter_spatial_11810506.tif", bbox_inches="tight",
                pad_inches=0.0)
    return 0


def sobel_filter_frequency(input_image):
    img = cv2.imread(input_image, 0)
    h = img.shape[0]
    w = img.shape[1]
    image_new = np.zeros([2 * h, 2 * w])
    image_new[0:h, 0:w] = img

    I, J = np.ogrid[0:2 * h, 0:2 * w]
    mask = np.full((2 * h, 2 * w), -1)
    mask[(I + J) % 2 == 0] = 1
    image_new = mask * image_new

    image_new_fft = np.fft.fft2(image_new)

    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    DFT_kernel = np.pad(kernel, (
        ((image_new.shape[0] - kernel.shape[0] + 1) // 2, (image_new.shape[0] - kernel.shape[0]) // 2),
        ((image_new.shape[1] - kernel.shape[1] + 1) // 2, (image_new.shape[1] - kernel.shape[1]) // 2)), mode='constant', constant_values=0)

    row, col = DFT_kernel.shape
    I, J = np.ogrid[0:row, 0:col]
    mask = np.full((row, col), -1)
    mask[(I + J) % 2 == 0] = 1
    DFT_kernel = mask * DFT_kernel

    DFT_kernel_fft = np.fft.fft2(DFT_kernel)

    output_frequency = np.real(np.fft.ifft2(image_new_fft * DFT_kernel_fft))

    row, col = output_frequency.shape
    I, J = np.ogrid[:row, :col]
    mask = np.full((row, col), -1)
    mask[(I + J) % 2 == 0] = 1
    output_frequency = mask * output_frequency

    # calculation error!!!!! -1 is important!!!!!!!!
    output_frequency = output_frequency[int(row / 2)-1:row-1, int(col / 2)-1:col-1]

    a = np.min(output_frequency)
    b = 255 / (np.max(output_frequency)-a)
    for i in range(0, h):
        for j in range(0, w):
            output_frequency[i, j] = int((output_frequency[i, j] - a)*b)
    plt.imshow(output_frequency, cmap='gray')
    plt.axis('off')
    plt.savefig(input_image.strip(".tif") + " sobel_filter_frequency_11810506.tif", bbox_inches="tight",
                pad_inches=0.0)
    return 0


sobel_filter_spatial("Q5_1.tif")
sobel_filter_frequency("Q5_1.tif")
