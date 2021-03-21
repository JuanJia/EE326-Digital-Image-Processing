import cv2
from skimage import io


def denoise(input_image):
    img = io.imread(input_image)
    img = cv2.fastNlMeansDenoising(img, 10,10,7,21)
    io.imsave(input_image.strip(".tif") + " denoise.tif", img)
    return input_image.strip(".tif") + " denoise.tif"
