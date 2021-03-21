from lab4.denoise import denoise
from lab4.gradient_sharpen import gradient_sharpen_robert_filter_abs, gradient_sharpen_robert_filter_sqrt, \
    gradient_sharpen_sobel_filter_abs, gradient_sharpen_sobel_filter_sqrt
from lab4.highBoost_sharpen import highBoost_sharpen
from lab4.laplace_sharpen import laplace_sharpen
from lab4.unsharpenedMasking_sharpen import unsharpenedMasking_sharpen

name = "Q4_1.tif"
gradient_sharpen_robert_filter_abs(name)
gradient_sharpen_robert_filter_sqrt(name)
gradient_sharpen_sobel_filter_abs(name)
gradient_sharpen_sobel_filter_sqrt(name)
highBoost_sharpen(name, 5)
laplace_sharpen(name, -1, 1)
laplace_sharpen(name, -1, 2)
unsharpenedMasking_sharpen(name)

name = denoise("Q4_2.tif")
gradient_sharpen_robert_filter_abs(name)
gradient_sharpen_robert_filter_sqrt(name)
gradient_sharpen_sobel_filter_abs(name)
gradient_sharpen_sobel_filter_sqrt(name)
highBoost_sharpen(name, 5)
laplace_sharpen(name, -1, 1)
laplace_sharpen(name, -1, 2)
unsharpenedMasking_sharpen(name)
