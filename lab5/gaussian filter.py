import matplotlib.pyplot as plt
import numpy as np
import cv2


def GaussianHighFilter(image_input, d):
    img = cv2.imread(image_input, 0)
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    s1 = np.log(np.abs(f_shift))

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(img.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transfor_matrix[i, j] = 1 - np.exp(-(dis ** 2) / (2 * (d ** 2)))
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * d_matrix)))
    return new_img


def GaussianLowFilter(image_input, d):
    img = cv2.imread(image_input, 0)
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    s1 = np.log(np.abs(f_shift))

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(img.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transfor_matrix[i, j] = np.exp(-(dis ** 2) / (2 * (d ** 2)))
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * d_matrix)))
    return new_img


plt.subplot(131)
plt.axis("off")
plt.imshow(GaussianHighFilter("Q5_2.tif", 30), cmap="gray")
plt.title('Q5_2 30')
plt.subplot(132)
plt.axis("off")
plt.title('Q5_2 60')
plt.imshow(GaussianHighFilter("Q5_2.tif", 60), cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("Q5_2 160")
plt.imshow(GaussianHighFilter("Q5_2.tif", 160), cmap="gray")
plt.axis('off')
plt.savefig("GaussianHighFilter_11810506.tif", bbox_inches="tight", pad_inches=0.0)
plt.show()
plt.close()

plt.subplot(131)
plt.axis("off")
plt.imshow(GaussianLowFilter("Q5_2.tif", 30), cmap="gray")
plt.title('Q5_2 30')
plt.subplot(132)
plt.axis("off")
plt.title('Q5_2 60')
plt.imshow(GaussianLowFilter("Q5_2.tif", 60), cmap="gray")
plt.subplot(133)
plt.axis("off")
plt.title("Q5_2 160")
plt.imshow(GaussianLowFilter("Q5_2.tif", 160), cmap="gray")
plt.axis('off')
plt.savefig("GaussianLowFilter_11810506.tif", bbox_inches="tight", pad_inches=0.0)
plt.show()
plt.close()