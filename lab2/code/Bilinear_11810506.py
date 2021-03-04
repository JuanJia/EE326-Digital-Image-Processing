from skimage import io,data
import numpy as np


def Bilinear_11810506(input_file, dim):
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

            x = i / (out_height-1) * (in_height-1)
            y = j / (out_width-1) * (in_width-1)

            x1 = int(x)
            y1 = int(y)

            x2 = x1
            y2 = y1 + 1

            x3 = x1 + 1
            y3 = y1

            x4 = x1 + 1
            y4 = y1 + 1

            u = x - x1
            v = y - y1

            # no out of bounds
            if x4 >= in_height:
                x4 = in_height - 2
                x1 = x1
                x2 = x2
                x3 = x4
            if y4 >= in_width:
                y4 = in_width - 2
                y1 = y1
                y2 = y4
                y3 = y3

            # interpolation
            out_image[i, j] = (1 - u) * (1 - v) * (in_image[x1, y1]) + (1 - u) * v * (in_image[x2, y2]) + u * (1 - v) * (in_image[x3, y3]) + u * v * (in_image[x4, y4])

    # Save Image
    if (out_width > in_width):
        io.imsave("Enlarged_Bilinear_11810506.tif", out_image)
    else:
        io.imsave("Shrunken_Bilinear_11810506.tif", out_image)


enlarged = round(256 * (1 + 6 / 10))
shrunken = round(256 * 6 / 10)

Bilinear_11810506("rice.tif", [enlarged, enlarged])
Bilinear_11810506("rice.tif", [shrunken, shrunken])