from skimage import io,data
import numpy as np
import cv2 as cv
from scipy.interpolate import interp2d


def Bicubic_11810506(input_file, dim):
    # Load image
    in_image = io.imread(input_file)
    # Dimension of output image
    out_width = dim[0]
    out_height = dim[1]
    # Dimension of input image
    in_width = in_image.shape[0]
    in_height = in_image.shape[1]
    # Initial the output image
    out_image = np.zeros(dim, dtype=np.uint8)

    # Perform Exchange
    out_x = np.linspace(0, in_height - 1, num=out_height)
    out_y = np.linspace(0, in_width - 1, num=out_width)
    f = interp2d(range(in_height), range(in_width), in_image, kind='cubic')
    out_image = f(out_x, out_y)

    # Save Image
    if (out_width > in_width):
        cv.imwrite("Enlarged_Bicubic_11810506.tif", out_image.astype(np.uint8))
    else:
        cv.imwrite("Shrunken_Bicubic_11810506.tif", out_image.astype(np.uint8))


enlarged = round(256 * (1 + 6 / 10))
shrunken = round(256 * 6 / 10)

Bicubic_11810506("rice.tif", [enlarged, enlarged])
Bicubic_11810506("rice.tif", [shrunken, shrunken])