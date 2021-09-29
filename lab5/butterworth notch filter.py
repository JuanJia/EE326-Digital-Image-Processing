import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.ticker import MultipleLocator


def butterworthPassFilter(image_input, sigma, n):
    image_shift = np.fft.fftshift(np.fft.fft2(cv2.imread(image_input, 0)))

    center_point = [[124, 84]]

    def make_transform_matrix(center_p):
        transform_matrix = np.zeros(image_shift.shape)
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    distance = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return distance

                dis = cal_distance(center_p, (i, j))
                transform_matrix[i, j] = (1 / (1 + (dis / sigma) ** (2 * n)))
        return transform_matrix

    mask = np.ones(image_shift.shape)
    for center in center_point:
        mask = mask * make_transform_matrix(center)

    output_frequency = np.multiply(image_shift, mask)

    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(output_frequency)))

    return new_img, mask * 255,np.log(np.abs(output_frequency))


img = cv2.imread("Q5_3.tif", 0)
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)
s1 = np.log(np.abs(f_shift))
plt.imshow(s1)
x_major_locator = MultipleLocator(20)
y_major_locator = MultipleLocator(20)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.tick_params(labelsize=4)
plt.grid()
plt.savefig("frequency 0", bbox_inches="tight",
            pad_inches=0.0, dpi=1000)
plt.close()

i = 1
for sigma in [10, 30, 50, 70]:
    for n in [1, 2, 3, 4]:
        plt.subplot(8, 6, i)
        plt.imshow(butterworthPassFilter("Q5_3.tif", sigma, n)[0], cmap="gray")
        plt.title("sigma=" + str(sigma) + ",n=" + str(n), fontsize=2,pad=2)
        plt.axis("off")

        plt.subplot(8, 6, i + 1)
        plt.imshow(butterworthPassFilter("Q5_3.tif", sigma, n)[1], cmap="gray")
        plt.title("sigma=" + str(sigma) + ",n=" + str(n), fontsize=2,pad=2)
        plt.axis("off")

        plt.subplot(8, 6, i + 2)
        plt.imshow(butterworthPassFilter("Q5_3.tif", sigma, n)[2], cmap="gray")
        plt.title("sigma=" + str(sigma) + ",n=" + str(n), fontsize=2, pad=2)
        plt.axis("off")
        i = i + 3
plt.savefig("butterworthPassFilter_11810506.tif", bbox_inches="tight", pad_inches=0.0, dpi=5000)
plt.close()
