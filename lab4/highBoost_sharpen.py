import cv2 as cv
import numpy as np


def highBoost_sharpen(input_image, k):
    src = cv.imread(input_image)
    # sigma = 5、15、25
    blur_img = cv.GaussianBlur(src, (0, 0), 5)
    usm = cv.addWeighted(src, 1 + k, blur_img, -k, 0)
    result = usm
    cv.imwrite(input_image.strip(".tif") + " highBoost_sharpen_11810506.tif", result)


