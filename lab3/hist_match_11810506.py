from typing import io
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_match_11810506(input_image, goal_image):
    # load original image
    img = cv2.imread(input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # load goal image
    goal_img = cv2.imread(goal_image)
    goal_img = cv2.cvtColor(goal_img, cv2.COLOR_BGR2GRAY)
    img_output = np.zeros(img.shape,dtype=np.uint8)
    input_histogram = []
    input_CDF = []
    input_CDF_scaled = []
    goal_histogram = []
    goal_CDF = []
    goal_CDF_scaled = []
    output_histogram = []
    output_CDF = []

    # equalization of goal image to obtain wanted histogram
    r, c = goal_img.shape
    for i in range(256):
        goal_histogram.append(np.sum(goal_img == i) / (r * c))
    sum = 0
    for i in range(len(goal_histogram)):
        sum = sum + goal_histogram[i]
        goal_CDF.append(sum)
    for i in range(256):
        goal_CDF_scaled.append(round(255 * goal_CDF[i]))

    # do equalizaiton to origianl image
    r, c = img.shape
    for i in range(256):
        input_histogram.append(np.sum(img == i) / (r * c))
    sum = 0
    # CDF
    for i in range(len(input_histogram)):
        sum = sum + input_histogram[i]
        input_CDF.append(sum)
    # integer part
    for i in range(256):
        input_CDF_scaled.append(round(255 * input_CDF[i]))

    g = []
    for i in range(256):
        s = input_CDF_scaled[i]
        flag = True
        for j in range(256):
            if goal_CDF_scaled[j] == s:
                g.append(j)
                flag = False
                break
        if flag == True:
            minp = 255
            jmin = 0
            for j in range(256):
                b = abs(goal_CDF_scaled[j] - s)
                if b < minp:
                    minp = b
                    jmin = j
            g.append(jmin)

    for i in range(r):
        for j in range(c):
            img_output[i, j] = g[img[i, j]]

    # equalization of output image to obtain output histogram
    r, c = img_output.shape
    for i in range(256):
        output_histogram.append(np.sum(img_output == i) / (r * c))
    sum = 0
    # CDF
    for i in range(len(output_histogram)):
        sum = sum + output_histogram[i]
        output_CDF.append(sum)

    io.imsave("Q3_2" + "_11810506.tif", img_output)
    n = np.arange(256)
    plt.plot(n, input_histogram)
    plt.savefig("Q3_2" + "_hist_11810506.tif")
    plt.close()
    plt.plot(n, goal_histogram)
    plt.savefig("Q3_2" + "_goal_hist_11810506.tif")
    plt.close()
    plt.plot(n, output_histogram)
    plt.savefig("Q3_2" + "_output_hist_11810506.tif")
    plt.close()
    return "Q3_2" + "_11810506.tif", "Q3_2" + "_hist_11810506.tif", "Q3_2" + "_goal_hist_11810506.tif", "Q3_2" + \
           "_output_hist_11810506.tif "


hist_match_11810506("Q3_2.tif", "Q3_2_goal.tif")
