from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from matplotlib.lines import Line2D
from operator import is_not
from functools import partial
import time
import glob

INPUT_PATH = "C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\LineTest\\"
PLANT_OUTPUT_PATH = "C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\boundary_lines_grid\\"
PVCFRAME_OUTPUT_PATH = "C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\bitwise_slic_pvc\\"
SUBTRACTED_IMAGE_OUTPUT_PATH = "C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\subtracted_output\\"

ORIGINAL_IMAGE_PATH = "C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\Fieldbook groundcover sample images\\"


for f in glob.glob(INPUT_PATH + "*.jpg"):
    print(f)
    img = io.imread(INPUT_PATH + f.split("\\")[-1])

    #img = io.imread("C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\HSV-bitwise.jpg")

    imgCopy = img.copy()

    t0 = time.time()

    labels1 = segmentation.slic(img, compactness=300, n_segments=1000, enforce_connectivity = False)

    t1 = time.time()

    print(t1 - t0)

    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

    cv2.imwrite(PLANT_OUTPUT_PATH + f.split("\\")[-1], out1)

    maskImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    pvcFrameImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    labelIds = np.unique(labels1)

    centers_labels = {}
        

    for i in labelIds:
        centers_labels[i] = np.mean(np.nonzero(labels1==i),axis=1)

    # centers
    # centers = np.array([np.mean(np.nonzero(labels1==i),axis=1) for i in labelIds])

    plantLocations = []
    
    cv2.imwrite("out1.jpg", out1)

#     pvcframeLocations = []
#     for k, v in centers_labels.items():
#         imgColor = out1[int(v[0]), int(v[1])]
#         if imgColor[0] > 80 and imgColor[1] > 0 and imgColor[2] > 80:
#             pvcframeLocations.append(k)    
        
    plantLocations = list(map(lambda x: x[0], centers_labels.items()))

    #plantLocations = list(map(lambda x: x[0] if (np.argmax(out1[int(x[1][0]), int(x[1][1])]) == 1 and np.max(out1[int(x[1][0]), int(x[1][1])]) > 0) else None, centers_labels.items()))

    plantLocations = list(filter(partial(is_not, None), plantLocations))

    #pvcframeLocations = list(filter(partial(is_not, None), pvcframeLocations))

    for i in plantLocations:
        lab = np.where(labels1 == i)
        for val in range(len(lab[0])):
            cv2.circle(maskImg, (lab[1][val], lab[0][val]), radius = 0, color = (255, 255, 255), thickness = -1)

#     for i in pvcframeLocations:
#         lab = np.where(labels1 == i)
#         for val in range(len(lab[0])):
#             cv2.circle(pvcFrameImg, (lab[1][val], lab[0][val]), radius = 0, color = (255, 255, 255), thickness = -1)


    cv2.imwrite(PLANT_OUTPUT_PATH + f.split("\\")[-1], maskImg) 

    cv2.imwrite(PVCFRAME_OUTPUT_PATH + f.split("\\")[-1], pvcFrameImg)

    originalImg = cv2.imread(INPUT_PATH + f.split("\\")[-1])

    maskImg = cv2.imread(PLANT_OUTPUT_PATH + f.split("\\")[-1], 0)

    bitwise = cv2.bitwise_and(originalImg, originalImg, mask = maskImg)

    lower_range = np.array([0, 0, 0])
    upper_range = np.array([170, 255, 255])

    hsv = cv2.cvtColor(bitwise, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, lower_range, upper_range)

    bitwise_and_img = cv2.bitwise_and(bitwise, bitwise, mask=thresh)

    cv2.imwrite("hsv.jpg", bitwise_and_img)

    lowerRange = np.array([0, 0, 0])
    upperRange = np.array([30, 255, 255])

    hsv = cv2.cvtColor(bitwise_and_img, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, lowerRange, upperRange)

    bitwise_and_img2 = cv2.bitwise_and(bitwise_and_img, bitwise_and_img, mask = thresh)

    cv2.imwrite("final_hsv.jpg", bitwise_and_img2)

    subImg = cv2.subtract(bitwise_and_img, bitwise_and_img2)

    cv2.imwrite(SUBTRACTED_IMAGE_OUTPUT_PATH + f.split("\\")[-1], subImg)
        
