import numpy as np
from skimage import io


def reduce_SAP_11810506(input_image, n_size):
    im = io.imread(input_image)
    step = int((n_size - 1) / 2)
    im_copy = np.zeros([im.shape[0] + 2 * step, im.shape[1] + 2 * step], dtype=np.uint8)

    for i in range(im_copy.shape[0]):
        for j in range(im_copy.shape[1]):
            if i < step or j < step or i >= im.shape[0]+step or j >= im.shape[1]+step:
                im_copy[i][j] = 0
            else:
                im_copy[i][j] = im[i - step][j - step]


    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i][j] = m_filter(i+step, j+step, n_size, im_copy)

    io.imshow(im_copy)
    io.imsave("Q3_4_11810506_" + str(n_size) + ".tif", im)
    return "Q3_4_11810506_" + str(n_size) + ".tif"


def m_filter(x, y, n_size, im):
    step = n_size
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(im[x + k][y + m])
            sum_s.sort()

    return sum_s[(int(step * step / 2) + 1)]


#for i in range(3, 19, 2):
    #reduce_SAP_11810506("Q3_4.tif", i)


#reduce_SAP_11810506("Q3_3_11810506.tif", 5)
reduce_SAP_11810506("Q3_4.tif", 3)