from skimage import io,data
import numpy as np


def Nearest_11810506(input_file, dim):
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
    for i in range(out_height):
        for j in range(out_width):
            # mapping
            m = round(i / (out_height-1) * (in_height-1))
            n = round(j / (out_width-1) * (in_width-1))
            # no out of bounds
            # interpolation
            out_image[i, j] = in_image[m, n]

    # Save image
    if (out_width > in_width):
        io.imsave("Enlarged_Nearest_11810506.tif", out_image)
    else:
        io.imsave("Shrunken_Nearest_11810506.tif", out_image)


enlarged = round(256 * (1 + 6 / 10))
shrunken = round(256 * 6 / 10)

Nearest_11810506("rice.tif", [enlarged, enlarged])
Nearest_11810506("rice.tif", [shrunken, shrunken])