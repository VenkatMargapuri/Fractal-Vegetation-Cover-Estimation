# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from math import sqrt
from itertools import combinations
from random import randint
import math
from collections import defaultdict, OrderedDict, Counter
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import cv2
from skimage.segmentation import mark_boundaries
from matplotlib.lines import Line2D
from operator import is_not, itemgetter
from functools import partial, reduce
import time
import glob
import os
from os import listdir
import random as rng
import glob
import csv
import pandas as pd
import sys
from shapely.geometry import LineString, Point
import operator

CURRENT_DIRECTORY = os.getcwd()
GRID_IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "grids", "") 
VERTICAL_LINE_INPUT_PATH = os.path.join(CURRENT_DIRECTORY, "PVCGrids", "")
SUBTRACTED_IMAGE_INPUT_PATH = os.path.join(CURRENT_DIRECTORY, "subtracted_output", "")
THRESHED_IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "threshed_images", "")
OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY, "boundary_lines_bitwise_slic_pvc_output", "")
REG_HOUGHLINES_PATH = os.path.join(CURRENT_DIRECTORY, "regHoughLines", "") 
CONTOUR_IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "PVCGrids", "")
CONTOUR_HORIZONTAL_LINE_INTERSECTION_PATH = os.path.join(CURRENT_DIRECTORY, "ContourHorizontalLineIntersections", "")
BITWISE_IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "bitwise", "")
CROPCOVER_IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "subtracted_output", "")
BITWISE_INPUT_PATH = os.path.join(CURRENT_DIRECTORY, "bitwise", "")
PLANT_OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY, "boundary_lines_grid", "")
PVCFRAME_OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY, "bitwise_slic_pvc", "")
FINAL_HSV_OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY, "finalHSV", "")
SUBTRACTED_IMAGE_OUTPUT_PATH = os.path.join(CURRENT_DIRECTORY, "subtracted_output", "")
UNWARPED_IMAGE_PATH = os.path.join(CURRENT_DIRECTORY, "OneSegmentUnwarped", "")

counter = 0
kernel = np.ones((3, 3), np.uint8)


# https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
# Merges multiple small line segments into a single line segment
def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 50
    min_angle_to_merge = 3

    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        #lines[idx] = False
            # append new group
            super_lines.append(new_group)


    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final

# Finds the length of a line segment given end points
def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

def merge_lines_segments1(lines, use_log=False):
    if(len(lines) == 1):
        return lines[0]

    line_i = lines[0]

    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):

        #sort by y
        points = sorted(points, key=lambda point: point[1])

        if use_log:
            print("use y")
    else:

        #sort by x
        points = sorted(points, key=lambda point: point[0])

        if use_log:
            print("use x")

    return [points[0], points[len(points)-1]]

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw
def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

    if (min(dist1,dist2,dist3,dist4) < 100):
        return True
    else:
        return False

#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
# http://paulbourke.net/geometry/pointlineplane/
def DistancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine


def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])


    return min(dist1,dist2,dist3,dist4)

# Finds the slope of the line given two points on it
def findSlope(a, b):
        if (b[0] - a[0]) == 0:
            return math.inf
        return ((b[1] - a[1])/(b[0] - a[0]))
    
# Finds the distance between vertical lines that are parallel to each other
# **** This method isn't used in the final version ****
def FindDistanceBetweenParallelVerticalLines(lines):
    validLines = []
    #if len(lines) == 1:
    #    return validLines.append((-1, 0))
    for i in range(0, len(lines) - 1):
        for j in range(1, len(lines)):
            (x2, y2) = lines[i][1]         
            (a2, b2) = lines[j][1]
            distance = abs(x2 - a2)
            # The distance between the two lines is expected to be 50px.
            # This is empirically determined and may be adjusted on an as-needed basis.
            if distance > 50:
                validLines.append((i, j))
    return validLines    

# Finds the longest line and magnitude from a list of lines
def GetLongestLineandMagnitude(lines):
    if len(lines) == 0:
        return -1
    maxMagnitude = -1
    longestLinedIndex = -1
    for i in range(len(lines)):
        currentMagnitude = lineMagnitude(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
        if currentMagnitude > maxMagnitude:
            longestLineIndex = i
            maxMagnitude = currentMagnitude
            
    #print("max magnitude: " + str(maxMagnitude))
    return lines[longestLineIndex], maxMagnitude


def GetVerticalandHorizontalLinesforGrid(horizontalLines, verticalLines, imgWidth, imgHeight):
    print("Hor Line: " + str(horizontalLines))
    print("Ver Line: " + str(len(verticalLines)))
    
    imgCenter = (imgHeight//2, imgWidth//2)

    
    if len(horizontalLines) == 0:
        horizontalLines.append([(0, 0), (imgWidth, 0)])
        horizontalLines.append([(0, imgHeight), (imgWidth, imgHeight)])
    elif len(horizontalLines) == 1:
        print("in horizontal line")
        line = horizontalLines[0]
        (x1, y1) = (line[0][0], line[0][1])
        if y1 < imgCenter[0]:
            horizontalLines.append([(0, imgHeight), (imgWidth, imgHeight)])
        else:
            horizontalLines.append([(0, 0), (imgWidth, 0)])
    
    if len(verticalLines) == 0:
        verticalLines.append([(0, 0), (imgHeight, 0)])
        verticalLines.append([(imgWidth, 0), (imgWidth, imgHeight)])
    elif len(verticalLines) == 1:
        line = verticalLines[0]
        (x1, y1) = (line[0][0], line[0][1])
        if x1 < imgCenter[1]:
            verticalLines.append([(imgWidth, 0), (imgWidth, imgHeight)])
        else:
            verticalLines.append([(0, 0), (0, imgHeight)])
    
    print(horizontalLines)
    leftLines = []
    rightLines = []
    topLines = []
    bottomLines = []
    for line in verticalLines:
        (x1, y1) = (line[0][0], line[0][1])
        (x2, y2) = (line[1][0], line[1][1])
        if x1 < imgCenter[1]:
            leftLines.append([(x1, y1), (x2, y2)])
        else:
            rightLines.append([(x1, y1), (x2, y2)])
            
    for line in horizontalLines:
        (x1, y1) = (line[0][0], line[0][1])
        (x2, y2) = (line[1][0], line[1][1])
        if y1 < imgCenter[0]:
            topLines.append([(x1, y1), (x2, y2)])
        else:
            bottomLines.append([(x1, y1), (x2, y2)])
            
    leftLines = sorted(leftLines, key = lambda x: x[0][0])
    rightLines = sorted(rightLines, key = lambda x: x[0][0])
    topLines = sorted(topLines, key = lambda x: x[0][1])
    bottomLines = sorted(bottomLines, key = lambda x: x[0][1])
    
    print("leftLines" + str(leftLines))
    print("rightLines" + str(rightLines))
    print("topLines" + str(topLines))
    print("bottomLines" + str(bottomLines))

    validLines = []
    if len(leftLines) > 0:
        validLines.append(leftLines[-1])
    else:
        validLines.append([(0, 0), (0, imgHeight)])
    
    if len(rightLines) > 0:
        validLines.append(rightLines[0])
    else:
        validLines.append([(imgWidth, 0), (imgWidth, imgHeight)])
        
    if len(topLines) > 0:
        validLines.append(topLines[-1])
    else:
        validLines.append([(0, 0), (imgWidth, 0)])
        
    if len(bottomLines) > 0:
        validLines.append(bottomLines[0])
    else:
        validLines.append([(0, imgHeight), (imgWidth, imgHeight)])

    print("valid lines length: " + str(len(validLines)))
        
    validLines = ExtendLinesInfinitely(validLines, imgHeight)          

    return validLines

def FindPointsofIntersectionofLines(lines, imgWidth, imgHeight):
    leftLine = validLines[0]
    rightLine = validLines[1]
    topLine = validLines[2]
    bottomLine = validLines[3] 
    
    print("points of intersection lines")
    print("left line: " + str(leftLine))
    print("right line: " + str(rightLine))
    print("top Line: " + str(topLine))
    print("bottom line: " + str(bottomLine))
    
    leftLine = LineString([(leftLine[0][0], leftLine[0][1]), (leftLine[1][0], leftLine[1][1])])
    rightLine = LineString([(rightLine[0][0], rightLine[0][1]), (rightLine[1][0], rightLine[1][1])])
    topLine = LineString([(topLine[0][0], topLine[0][1]), (topLine[1][0], topLine[1][1])])
    bottomLine = LineString([(bottomLine[0][0], bottomLine[0][1]), (bottomLine[1][0], bottomLine[1][1])])
    
    upperLeftIntersection = leftLine.intersection(topLine)
    lowerLeftIntersection = leftLine.intersection(bottomLine)
    upperRightIntersection = rightLine.intersection(topLine)
    lowerRightIntersection = rightLine.intersection(bottomLine)
    
    if upperLeftIntersection.is_empty:
        upperLeftIntersection = Point(0, 0)
    if lowerLeftIntersection.is_empty:
        lowerLeftIntersection = Point(0, imgHeight)
    if upperRightIntersection.is_empty:
        upperRightIntersection = Point(imgWidth, 0)
    if lowerRightIntersection.is_empty:
        lowerRightIntersection = Point(int(imgWidth - 1), int(imgHeight - 1))
    
    print("lower right intersection" + str(lowerRightIntersection))
    pts = np.array([[int(upperLeftIntersection.x), int(upperLeftIntersection.y)], [int(upperRightIntersection.x), int(upperRightIntersection.y)], 
                           [int(lowerRightIntersection.x), int(lowerRightIntersection.y)], [int(lowerLeftIntersection.x), int(lowerLeftIntersection.y)]])
    
    return pts

def UnwarpImage(pts, img):
    minRect = cv2.minAreaRect(pts)
    
    box = cv2.boxPoints(minRect)   
    box = np.int0(box) 
    
    width = int(minRect[1][0])
    height = int(minRect[1][1])
    
    src_pts = box.astype("float32")
    
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    
    cv2.imwrite("unwarped.jpg", warped)  
    
    return warped

def ExtractBitwiseImagesofPVCGrid(pts, img):
                
    grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    maskImg = np.zeros(grayscaleImg.shape, np.uint8)
            
    # Draws a polygon in white color around the segment using the boundary points identified
    maskImg = cv2.fillConvexPoly(maskImg, pts, color = 255)    
    
    rect = cv2.boundingRect(pts)
        
    cropped = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    
    #cv2.imwrite("polyMask.jpg", maskImg)
    #cv2.imwrite("cropped.jpg", cropped)
    
    return cropped, maskImg
        
def EstimatePlantCover(polygonalMask, plantCoverImg):
    whitePixels = np.sum(polygonalMask == 255)
    
    bitwise = cv2.bitwise_and(plantCoverImg, plantCoverImg, mask = polygonalMask)
    
    #cv2.imwrite("plantCover.jpg", bitwise)
    
    #RemoveGridfromSLICImage(bitwise)
    
    plantCover = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)
    
    #cv2.imwrite("plantCoverGray.jpg", plantCover)
    
    #print(plantCover)
    
    plantCover[np.where(plantCover >240)] = 0
    
    #print(np.amax(plantCover))
    
    ret, thresh = cv2.threshold(plantCover, 30, 255, cv2.THRESH_BINARY)
    
    plantCoverPixels = np.sum(thresh == 255)
    
    plantCoverPercentage = (plantCoverPixels/whitePixels) * 100
    
    return plantCoverPercentage
        
# Finds the vertical boundaries of the PVC grid i.e. the lines that pass from top to bottom along the boundary of the PVC grid
# **** This method is not used in the final version ****
def GetVerticalBoundariesofPVCGrid(lines, imgWidth):
    
    # Find the center of the image along the x axis
    # It helps determine if a line is to the left or right boundary of the PVC grid
    imgCenter = int(imgWidth//2)
    leftLines = []
    rightLines = []
    for line in lines:
        (x1, y1) = line[0]
        (x2, y2) = line[1]
        if x1 < imgCenter:
            leftLines.append(line)
        else:
            rightLines.append(line)
      
    # Find the distance between the lines along the left and right boundaries
    # The lines which satisfy the distance condition outlined in the 'FindDistanceBetweenParallelVerticalLines' function 
    # are deemed valid lines.
    leftValidLines = FindDistanceBetweenParallelVerticalLines(leftLines)
    rightValidLines = FindDistanceBetweenParallelVerticalLines(rightLines)
    
    leftEdgeLines = []
    line_mins ={}

    for i in range(len(leftValidLines)):
        leftEdgeLines.append((leftLines[leftValidLines[i][0]], leftLines[leftValidLines[i][1]]))
    
    if len(leftEdgeLines) > 1:
        for i in range(len(leftEdgeLines)):
            (x1, y1) = leftEdgeLines[i][0][0]
            (x2, y2) = leftEdgeLines[i][1][0]
            line_mins[i] = min(x1, x2)
            
    # Sort the list of lines so as to identify if the line is the inner or outer line.
    # The idea is that the PVC grid has two boundary lines and it is required to identify if a line is either the inner or outer boundary line
    line_mins = sorted(line_mins.items(), key = lambda x: x[1])
    
    
    if len(line_mins) > 0:
        leftEdgeLines = leftEdgeLines[line_mins[0][0]]
    elif len(leftEdgeLines) == 1:        
        leftEdgeLines = leftEdgeLines[0]
        
    rightEdgeLines = []
    line_max = {}
    for i in range(len(rightValidLines)):
        rightEdgeLines.append((rightLines[rightValidLines[i][0]], rightLines[rightValidLines[i][1]]))
        
    if len(rightEdgeLines) > 1:
        for i in range(len(rightEdgeLines)):
            (x1, y1) = rightEdgeLines[i][0]
            (x2, y2) = rightEdgeLines[i][1]
            line_max[i] = max(x1, x2)
            
    # Sort the list of lines so as to identify if the line is the inner or outer line.
    # The idea is that the PVC grid has two boundary lines and it is required to identify if a line is either the inner or outer boundary line
    line_max = sorted(line_max.items(), key = lambda x: x[1])
    
    if len(line_max) > 0:
        rightEdgeLines = rightEdgeLines[line_max[0][0]]
    elif len(rightEdgeLines) == 1:
        rightEdgeLines = rightEdgeLines[0]
        
    mergedList = []
    mergedList.extend(leftEdgeLines)
    mergedList.extend(rightEdgeLines)
    
    return mergedList

# Checks is the line is on the grid
# **** This method is not used in the final version ****
def CheckifLineisonGrid(line, img):
    im = img.copy()
    gridPixels = 0
    numPoints = int(lineMagnitude(line[0][0], line[0][1], line[0][2], line[0][3]))//4
    for p in np.linspace(np.array([line[0][0], line[0][1]]), np.array([line[0][2], line[0][3]]), numPoints):        
        if img[int(p[1]), int(p[0])] > 0:
            gridPixels = gridPixels + 1
            cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)
    if gridPixels//numPoints > 0.5:
        return True
    return False

# From a list of lines, finds the line closest to the boundary and is atleast the percent of magnitude of the longest line within the list of lines
def FindLineClosesttoBoundary(lines, maxMagnitude, percentMagnitude):
    lineIndex = -1
    for i in range(len(lines)):
        if lineMagnitude(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]) > percentMagnitude * maxMagnitude:
            lineIndex = i
            break
    return lineIndex
    

# Finds the vertical lines along the left and right boundaries of the PVC grid
def FindLeftandRightVerticalLinesAlongGrid(verticalLines, img, imgWidth):
    imgMidPoint = imgWidth//2
    selectedLines = []
    leftVerticalLines = []
    rightVerticalLines = []
    
    # Checks to see if the vertical line is to the right or left side of the image
    for line in verticalLines:
        if line[0][0] < imgMidPoint:
            leftVerticalLines.append(line)
        elif line[0][0] > imgMidPoint:
            rightVerticalLines.append(line)
                        
    # Retrieves the longest line and magnitude from list of lines
    leftLongestVerticalLine, maxLeftMagnitude = GetLongestLineandMagnitude(leftVerticalLines)
    rightLongestVerticalLine, maxRightMagnitude = GetLongestLineandMagnitude(rightVerticalLines)
    
    # Sort the list of left and right vertical lines by the X co-ordinate.
    # Sorting is done both in ascending and descending order to help with identifying the inner and outer lines along the boundary.
    ascendingLeftVerticalLines = sorted(leftVerticalLines, key = lambda x: x[0][0])
    descendingLeftVerticalLines = sorted(leftVerticalLines, key = lambda x: x[0][0], reverse = True)    
    ascendingRightVerticalLines = sorted(rightVerticalLines, key = lambda x: x[0][0])
    descendingRightVerticalLines = sorted(rightVerticalLines, key = lambda x: x[0][0], reverse = True)    
    
    leftLineIndex = -1
    rightLineIndex = -1
    
    # Find the index of the outer line at the left boundary
    leftLineIndex = FindLineClosesttoBoundary(ascendingLeftVerticalLines, maxLeftMagnitude, 0.9)
            
    # Find the index of the outer line at the right boundary
    rightLineIndex = FindLineClosesttoBoundary(descendingRightVerticalLines, maxRightMagnitude, 0.9) 
                            
    leftLine = ascendingLeftVerticalLines[leftLineIndex]
    rightLine = descendingRightVerticalLines[rightLineIndex]
        
    selectedLines.append(leftLine)
    selectedLines.append(rightLine)    
  
    # Find the index of the inner line at the left boundary
    leftLineIndex = FindLineClosesttoBoundary(descendingLeftVerticalLines, maxLeftMagnitude, 0.9)
    
    # Find the index of the inner line at the right boundary
    rightLineIndex = FindLineClosesttoBoundary(ascendingRightVerticalLines, maxLeftMagnitude, 0.9)
            
    leftLine = descendingLeftVerticalLines[leftLineIndex]
    rightLine = ascendingRightVerticalLines[rightLineIndex]
    
    selectedLines.append(leftLine)
    selectedLines.append(rightLine)
    
    # Sort the selected lines so the lines are arranged in order starting with the left-most line 
    selectedLines = sorted(selectedLines, key = lambda x: x[0][0])

    return selectedLines              


        
# Assigns the detected hough lines to each of the segments in the PVC grid.
# Checks if each segment has two lines or more. 
# Sometimes, there are more than two lines assigned where some ofthe lines are usually overlapping one another.
# This method ensures that there are only two lines assigned to each segment by filtering out the overlapping lines
def AssignLinestoEachSegment(rect_lines, img, bitImgShape):    
    for k, v in rect_lines.items():  
        segment = k       
        
        if len(v) == 2:
            line_angles = {}
            line_angles[0] = []
            line_angles[1] = []
            
            line_positions = {}
            for i in range(len(v)):
                (x1, y1), (x2, y2) = v[i]
                line_angles[i] = (findSlope((x1, y1), (x2, y2)))
                line_positions[i] = y2 if y1 < y2 else y1
                
            line_angles = sorted(line_angles.items(), key = lambda x: abs(x[1]))    
            line_positions = sorted(line_positions.items(), key = lambda x: x[1])

            # Checks to see if teh two lines within the segment are apart by a certain threshold.
            # The value 2.5 is empirically determined and may be adjusted on an as-needed basis.
            angleDiff = reduce(lambda x, y: math.degrees(math.atan(abs((x[1] - y[1])/(1 + (x[1] * y[1]))))), line_angles)
            print("angleDiff: " + str(angleDiff))
            
            # In case the two lines are apart by over 2.5 degrees, the line that is most oriented is adjusted to be parallel to the line with lowest orientation
            # The position of the most angled line within the segment i.e. is it the top or bottom line and its slope are the two factors considered to make the adjustment.
            if(abs(angleDiff) > 2.5):                  
                MostAngledLineIndex = list(line_angles)[-1]    
                
                otherLineIndex = list(line_angles)[0]
                
                (x1, y1), (x2, y2) = v[MostAngledLineIndex[0]]
                
                (a1, b1), (a2, b2) = v[otherLineIndex[0]]

                IsMostAngledLineAtTop = True if list(line_positions)[MostAngledLineIndex[0]] == 0 else False
                
                MostAngledLineSlope = findSlope((x1, y1), (x2, y2))
                
                otherLineSlope = findSlope((a1, b1), (a2, b2))                

                if not IsMostAngledLineAtTop:
                    y1Greater = True if y1 > y2 else False

                    if y1Greater:                        
                        y2 = int(y1 + ((x2 - x1) * otherLineSlope))
                        v[MostAngledLineIndex[0]] = [(x1, y1), (x2, y2)]                        
                    else:
                        y1 = int(y2 - ((x2 - x1) * otherLineSlope))
                        v[MostAngledLineIndex[0]] = [(x1, y1), (x2, y2)]                
                else:
                    y1Lesser = True if y1 < y2 else False
                    if y1Lesser:
                        y2 = int(y1 + ((x2 - x1) * otherLineSlope))
                        v[MostAngledLineIndex[0]] = [(x1, y1), (x2, y2)]
                    else:
                        y1 = int(y2 - ((x2 - x1) * otherLineSlope))                        
                        v[MostAngledLineIndex[0]] = [(x1, y1), (x2, y2)]
                rect_lines[k] = v                
        else:
            line_angles = {}
            line_positions = {}
            
            # Find the line orientation (angle) and position of each of the lines within the segment
            # Line position means the location of the greater of the y co-ordinates. This helps identify the location of the line on the image
            for i in range(len(v)):
                (x1, y1), (x2, y2) = v[i]
                line_angles[i] = (math.degrees(findSlope((x1, y1), (x2, y2))))   
                line_positions[i] = y2 if y1 < y2 else y1
           
            # The line angles and line positions dictionaries are sorted based on angles and positions respectively.
            line_angles = sorted(line_angles.items(), key = lambda x: abs(x[1])) 
            line_positions = sorted(line_positions.items(), key = lambda x: x[1])
            
            # Finds a pair of lines with the least orientation and separated by a distance of a certain threshold
            # The distance of 25 is determined empirically and may be adjusted on an as-needed basis.
            for i in range(len(line_angles) - 1):
                for j in range(i + 1, len(line_angles)):
                    for l, s in line_positions:
                        if line_angles[i][0] == l:
                            y1 = s
                            break
                            
                    for l, s in line_positions:
                        if line_angles[j][0] == l:
                            y2 = s
                            break
                            
                    if(abs(y2 - y1) >= 25):
                        chosenLines = []
                        chosenLines.append(v[line_angles[i][0]])
                        chosenLines.append(v[line_angles[j][0]]) 
                        break
                break
            line_angles = {}
            line_angles[0] = []
            line_angles[1] = []
            
            line_positions = {}

            for i in range(len(chosenLines)):
                (x1, y1), (x2, y2) = chosenLines[i]
                line_angles[i] = (findSlope((x1, y1), (x2, y2)))
                line_positions[i] = y2 if y1 < y2 else y1
                
            line_angles = sorted(line_angles.items(), key = lambda x: abs(x[1]))    
            line_positions = sorted(line_positions.items(), key = lambda x: x[1])
            
            angleDiff = reduce(lambda x, y: math.degrees(math.atan(abs((x[1] - y[1])/(1 + (x[1] * y[1]))))), line_angles)

            angleDiff = abs(angleDiff)
            
            if(angleDiff > 2.5):          
                MostAngledLineIndex = line_angles[-1] 

                (x1, y1), (x2, y2) = chosenLines[MostAngledLineIndex[0]]
                
                IsMostAngledLineAtTop = True if list(line_positions)[MostAngledLineIndex[0]] == 0 else False
                
                MostAngledLineSlope = findSlope((x1, y1), (x2, y2))

                if not IsMostAngledLineAtTop:
                    
                    y1Greater = True if y1 > y2 else False
                    
                    if y1Greater:
                        chosenLines[MostAngledLineIndex[0]] = [(x1, y1), (x2, y1)]                        
                    else:
                        chosenLines[MostAngledLineIndex[0]] = [(x1, y2), (x2, y2)]                
                else:
                    y1Lesser = True if y1 < y2 else False
                    if y1Lesser:
                        chosenLines[MostAngledLineIndex[0]] = [(x1, y1), (x2, y1)]
                    else:
                        chosenLines[MostAngledLineIndex[0]] = [(x1, y2), (x2, y2)]
                        
                rect_lines[k] = chosenLines
            else:
                rect_lines[k] = chosenLines#[v[line_angles[i][0]], v[line_angles[j][0]]]    
                    
    rect_lines = FindIntersectionofHorizontalLineswithGridContour(rect_lines, img)

    rect_lines_list = [v for k, v in rect_lines.items()]

    return rect_lines
    
def WriteImage(img, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    cv2.imwrite(os.path.join(CURRENT_DIRECTORY, dir, filename.split("\\")[-1]), img)    

# Finds the intersection of the horizontal lines with grid contour
def FindIntersectionofHorizontalLineswithGridContour(rect_lines, contourImg):
    count = 0
    ctrImg = contourImg.copy()
    for k, v in rect_lines.items():
        currentSegment = k
        newLines = []        
        
        # Extends the horizontal lines infinitely (by a factor of 10000) so they intersect with the grid contour
        for line in v:
            tempImg = np.zeros((contourImg.shape[0], contourImg.shape[1]), np.uint8)            
            tempImg2 = np.zeros((contourImg.shape[0], contourImg.shape[1]), np.uint8)            
            slope = (findSlope(line[0], line[1]))
            r = math.sqrt(1 + (slope * slope))     
            length = lineMagnitude(line[0][0], line[0][1], line[1][0], line[1][1])
            
            x1 = int(line[0][0] - 10000 * math.cos(slope))
            y1 = int(line[0][1] - 10000 * math.sin(slope))
            
            x2 = int(line[1][0] + 10000 * math.cos(slope))
            y2 = int(line[1][1] + 10000 * math.sin(slope))
            
            tempImg = cv2.line(tempImg, (x1, y1), (x2, y2), 255, 1)
            tempImg2 = cv2.line(ctrImg, (x1, y1), (x2, y2), 255, 1)
            
            bitwise = cv2.bitwise_and(tempImg, tempImg, mask = contourImg)
            
            bitwise[np.array(bitwise) > 0] = 255
            intersectionPoints = np.where(bitwise > 0)
            
            yIntersection = intersectionPoints[0]
            xIntersection = intersectionPoints[1]
                        
            if len(xIntersection) == 2:
                newLines.append([(xIntersection[0], yIntersection[0]), (xIntersection[1], yIntersection[1])])
            
            if len(xIntersection) > 2:
                intersection = list(zip(xIntersection, yIntersection))
                intersection.sort(key = lambda x: x[0])
                newLines.append([intersection[0], intersection[-1]])
       

            cv2.imwrite(CONTOUR_HORIZONTAL_LINE_INTERSECTION_PATH + str(count * 50) + ".jpg" , bitwise)
            cv2.imwrite(CONTOUR_HORIZONTAL_LINE_INTERSECTION_PATH + str(count * 5) + ".jpg" , tempImg2)
            count = count + 1 
        if len(newLines) > 0:
            rect_lines[k] = newLines

    return rect_lines

# Finds the area of each segment of the PVC grid in px
# **** This method is not used in the final version ****
def FindAreaofEachSegment(rect_lines, verticalLines):
    segmentAreas = {}
    maxSegment = max(k for k, v in rect_lines.items())

    for k, v in rect_lines.items():
        if k <= maxSegment - 1:
            upperSegmentLines = rect_lines[k]
            lowerSegmentLines = rect_lines[k + 1]
            
            (x1, y1), (x2, y2) = upperSegmentLines[0]
            (a1, b1), (a2, b2) = upperSegmentLines[1]
            
            if y1 <= b1:
                upperSegmentLowerLine = [(a1, b1), (a2, b2)]
                upperSegmentUpperLine = [(x1, y1), (x2, y2)]
            else:
                upperSegmentLowerLine = [(x1, y1), (x2, y2)]
                upperSegmentUpperLine = [(a1, b1), (a2, b2)]
            
            (w1, e1), (w2, e2) = lowerSegmentLines[0]
            (q1, r1), (q2, r2) = lowerSegmentLines[1]
            
            if e1 <= r1:
                lowerSegmentUpperLine = [(w1, e1), (w2, e2)]
                lowerSegmentLowerLine = [(q1, r1), (q2, r2)]
            else:
                lowerSegmentUpperLine = [(q1, r1), (q2, r2)]
                lowerSegmentLowerLine = [(w1, e1), (w2, e2)]
                
            innerLeftVerticalLine = verticalLines[1]
            innerRightVerticalLine = verticalLines[2]
            
            #LineString([(validLines[line1][0][0], yvalue), (validLines[line2][0][0], yvalue)])
            print(str(k))
            innerLeftVerticalLine = LineString([(innerLeftVerticalLine[0][0], innerLeftVerticalLine[0][1]), (innerLeftVerticalLine[0][2], innerLeftVerticalLine[0][3])])
            innerRightVerticalLine = LineString([(innerRightVerticalLine[0][0], innerRightVerticalLine[0][1]), (innerRightVerticalLine[0][2], innerRightVerticalLine[0][3])])
            upperSegmentLowerLine = LineString(upperSegmentLowerLine)
            lowerSegmentUpperLine = LineString(lowerSegmentUpperLine)
            
            upperLeftIntersection = innerLeftVerticalLine.intersection(upperSegmentLowerLine)
            upperRightIntersection = innerRightVerticalLine.intersection(upperSegmentLowerLine)
            lowerLeftIntersection = innerLeftVerticalLine.intersection(lowerSegmentUpperLine)
            lowerRightIntersection = innerRightVerticalLine.intersection(lowerSegmentUpperLine)
            

            
            area1 = FindAreaofTriangle(upperLeftIntersection, lowerLeftIntersection, lowerRightIntersection)
            area2 = FindAreaofTriangle(upperLeftIntersection, upperRightIntersection, lowerRightIntersection)
            
            #print("area1 " + str(area1))
            #print("area2 " + str(area2))
            
            segmentArea = area1 + area2
            
            segmentAreas[k] = segmentArea
            
    return segmentAreas
    
        
# Estimates the crop cover within each segment of the PVC grid    
def EstimateCropCoverwithinSegment(rect_lines, verticalLines, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)    
    
    maxSegment = max(k for k, v in rect_lines.items())

    counter = 0
    for k, v in rect_lines.items():
        maskImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

        if k <= maxSegment - 1:
            # Find the lines belonging to the current segment and the subsequent segment.
            # Segment 0 is the uppermost segment of the PVC grid. So, we get segment (0, 1), then (1, 2) and so on.
            upperSegmentLines = rect_lines[k]
            lowerSegmentLines = rect_lines[k + 1]
            
            # Each segment has two lines within it since the boundary of the PVC grid is two lines.
            (x1, y1), (x2, y2) = upperSegmentLines[0]
            (a1, b1), (a2, b2) = upperSegmentLines[1]
            
            # Finds the upper and lower lines within each segment of the PVC grid
            if y1 <= b1:
                upperSegmentLowerLine = [(a1, b1), (a2, b2)]
                upperSegmentUpperLine = [(x1, y1), (x2, y2)]
            else:
                upperSegmentLowerLine = [(x1, y1), (x2, y2)]
                upperSegmentUpperLine = [(a1, b1), (a2, b2)]
            
            (w1, e1), (w2, e2) = lowerSegmentLines[0]
            (q1, r1), (q2, r2) = lowerSegmentLines[1]
            
            if e1 <= r1:
                lowerSegmentUpperLine = [(w1, e1), (w2, e2)]
                lowerSegmentLowerLine = [(q1, r1), (q2, r2)]
            else:
                lowerSegmentUpperLine = [(q1, r1), (q2, r2)]
                lowerSegmentLowerLine = [(w1, e1), (w2, e2)]
                
            # Retrieves the inner left and inner right vertical lines of the PVC grid.
            # The inner lines act as the boundaries of the segment.
            innerLeftVerticalLine = verticalLines[1]
            innerRightVerticalLine = verticalLines[2]
                        
            # Converts the horizontal and vertical boundary lines to LineStrings to be used in the 'intersection' function from Shapely
            innerLeftVerticalLine = LineString([(innerLeftVerticalLine[0][0], innerLeftVerticalLine[0][1]), (innerLeftVerticalLine[1][0], innerLeftVerticalLine[1][1])])
            innerRightVerticalLine = LineString([(innerRightVerticalLine[0][0], innerRightVerticalLine[0][1]), (innerRightVerticalLine[1][0], innerRightVerticalLine[1][1])])
            upperSegmentLowerLine = LineString(upperSegmentLowerLine)
            lowerSegmentUpperLine = LineString(lowerSegmentUpperLine)                        
            
            # Find the intersection of the lines
            upperLeftIntersection = innerLeftVerticalLine.intersection(upperSegmentLowerLine)
            upperRightIntersection = innerRightVerticalLine.intersection(upperSegmentLowerLine)
            lowerLeftIntersection = innerLeftVerticalLine.intersection(lowerSegmentUpperLine)
            lowerRightIntersection = innerRightVerticalLine.intersection(lowerSegmentUpperLine)
            
            #print("upper left intersection: " + str(upperLeftIntersection))
            #print("upper right intersection: " + str(upperRightIntersection))
            #print("lower left intersection: " + str(lowerLeftIntersection))
            #print("lower right intersection: " + str(lowerRightIntersection))
            
                        
            pts = np.array([[int(upperLeftIntersection.x), int(upperLeftIntersection.y)], [int(upperRightIntersection.x), int(upperRightIntersection.y)], 
                           [int(lowerRightIntersection.x), int(lowerRightIntersection.y)], [int(lowerLeftIntersection.x), int(lowerLeftIntersection.y)]])
            
            # Draws a polygon in white color around the segment using the boundary points identified
            maskImg = cv2.fillConvexPoly(maskImg, pts, color = 255)  
            
            # Counts the number of white pixels within the segment polygon 
            whitePixels = np.sum(maskImg == 255)
            
            print("segment: " + str(k))
            
            print("white: " +  str(whitePixels))
            
            # Applies the bitwise_and operation on the image containing crop cover and segment polygon
            bitwise = cv2.bitwise_and(thresh, thresh, mask = maskImg)
            
            # Counts the number of white pixels. The white pixels indicate the plant cover.
            plantPixels = np.sum(bitwise == 255)
            
            print("plant pixels: " + str(plantPixels))
            
            print("plant cover: " + str((plantPixels/whitePixels) * 100))
            cv2.imwrite("maskImage_" + str(counter) + ".jpg", maskImg)
            cv2.imwrite("plantCoverImage_" + str(counter) + ".jpg", bitwise)
            counter = counter + 1
                        
# Finds area of triangle formed by three points
# **** This method is not used in the final version ****
def FindAreaofTriangle(pt1, pt2, pt3):
    return 0.5 * abs((pt1.x * (pt2.y - pt3.y)) + (pt2.x * (pt3.y - pt1.y)) + (pt3.x * (pt1.y - pt2.y))) 

# Finds vertical lines along the PVC grid
# **** This method is not used in the final version ****
def FindVerticalLinesAlongGrid(rect_lines):
    verticalLines = []
    #print(rect_lines)
    maxSegment = max(k for k, v in rect_lines.items())
    #print(maxSegment)
    for k, v in rect_lines.items():
        if k <= maxSegment - 1:
            upperSegmentLines = rect_lines[k]
            lowerSegmentLines = rect_lines[k + 1]
            
            (x1, y1), (x2, y2) = upperSegmentLines[0]
            (a1, b1), (a2, b2) = upperSegmentLines[1]
            
            if y1 <= b1:
                upperSegmentLowerLine = [(a1, b1), (a2, b2)]
                upperSegmentUpperLine = [(x1, y1), (x2, y2)]
            else:
                upperSegmentLowerLine = [(x1, y1), (x2, y2)]
                upperSegmentUpperLine = [(a1, b1), (a2, b2)]
            
            (w1, e1), (w2, e2) = lowerSegmentLines[0]
            (q1, r1), (q2, r2) = lowerSegmentLines[1]
            
            if e1 <= r1:
                lowerSegmentUpperLine = [(w1, e1), (w2, e2)]
                lowerSegmentLowerLine = [(q1, r1), (q2, r2)]
            else:
                lowerSegmentUpperLine = [(q1, r1), (q2, r2)]
                lowerSegmentLowerLine = [(w1, e1), (w2, e2)]
                
            #print("segment: " + str(k))
            #print(upperSegmentUpperLine)
            #print(upperSegmentLowerLine)
            #print(lowerSegmentUpperLine)
            #print(lowerSegmentLowerLine)
            #print(abs(upperSegmentLowerLine[0][0] - lowerSegmentUpperLine[0][0]))
            #print(abs(upperSegmentUpperLine[0][0] - lowerSegmentUpperLine[0][0]))
            #print(abs(upperSegmentLowerLine[0][0] - lowerSegmentLowerLine[0][0]))
            #print(abs(upperSegmentUpperLine[0][0] - lowerSegmentLowerLine[0][0]))
            if abs(upperSegmentLowerLine[0][0] - lowerSegmentUpperLine[0][0]) < 300:
                verticalLines.append([(upperSegmentLowerLine[0][0], upperSegmentLowerLine[0][1]), (lowerSegmentUpperLine[0][0], lowerSegmentUpperLine[0][1])])
                verticalLines.append([(upperSegmentLowerLine[1][0], upperSegmentLowerLine[1][1]), (lowerSegmentUpperLine[1][0], lowerSegmentUpperLine[1][1])])
                break
            elif abs(upperSegmentUpperLine[0][0] - lowerSegmentUpperLine[0][0]) < 300:
                verticalLines.append([(upperSegmentUpperLine[0][0], upperSegmentUpperLine[0][1]), (lowerSegmentUpperLine[0][0], lowerSegmentUpperLine[0][1])])
                verticalLines.append([(upperSegmentUpperLine[1][0], upperSegmentUpperLine[1][1]), (lowerSegmentUpperLine[1][0], lowerSegmentUpperLine[1][0])])
                break
            elif abs(upperSegmentLowerLine[0][0] - lowerSegmentLowerLine[0][0]) < 300:
                verticalLines.append([(upperSegmentLowerLine[0][0], upperSegmentLowerLine[0][1]), (lowerSegmentLowerLine[0][0], lowerSegmentLowerLine[0][1])])
                verticalLines.append([(upperSegmentLowerLine[1][0], upperSegmentLowerLine[1][0]), (lowerSegmentLowerLine[1][0], lowerSegmentLowerLine[1][0])])
                break
            elif abs(upperSegmentUpperLine[0][0] - lowerSegmentLowerLine[0][0]) < 300:
                verticalLines.append([(upperSegmentUpperLine[0][0], upperSegmentUpperLine[0][1]), (lowerSegmentLowerLine[0][0], lowerSegmentLowerLine[0][1])])
                verticalLines.append([(upperSegmentUpperLine[1][0], upperSegmentUpperLine[1][1]), (lowerSegmentLowerLine[1][0], lowerSegmentLowerLine[1][1])])
                break
    return verticalLines
                                                    
    
    
def GroupLinesbySegment(lines, imgHeight, segmentNum):
    segmentHeight = imgHeight//segmentNum
    rect_lines = defaultdict(list)
    for i in range(len(lines)):
        found = False
        (x1, y1) = lines[i][0]
        startcounter = 0
        endcounter = 0.5
        while not found:
            if segmentHeight * startcounter < y1 < segmentHeight * endcounter:
                rect_lines[int(endcounter)].append(lines[i])
                found = True
            else:
                startcounter = endcounter
                endcounter = endcounter + 1
    return rect_lines

def RetrieveParallelLines(lines):
    lineDict = {}
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        slope = round(abs(findSlope((x1, y1), (x2, y2))), 2)
        lineDict[i] = slope
    mode = findDictMode(lineDict)
    #print("mode :" + str(mode))
    horizontalLines = []
    verticalLines = []
    for k, v in lineDict.items(): 
        if v < 10 and v != 0.45:
            horizontalLines.append(lines[k])  
        if v > 10:               
            verticalLines.append(lines[k])
           
    return horizontalLines, verticalLines

def findDictMode(input_dict):
    track={}

    for key,value in input_dict.items():
        if value not in track:
            track[value]=0
        else:
            track[value]+=1        

    return max(track,key=track.get)

def get_lines(lines_in):
    return [l[0] for l in lines_in]

def MergeLines(lines):
    # prepare
    _lines = []
    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
    # sort
    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
            _lines_y.append(line_i)
        else:
            _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)

    merged_lines_all = []
    merged_lines_all.extend(merged_lines_x)
    merged_lines_all.extend(merged_lines_y) 
    return merged_lines_all
    

def ExtendLinesInfinitely(lines, imgHeight):
    extendedLines = []
    
    for line in lines:
        slope = (findSlope(line[0], line[1]))
        
        if math.isinf(abs(slope)):
            y1 = 0
            y2 = imgHeight
            print("inf line: " + str(line))
            extendedLines.append([(line[0][0], y1), (line[1][0], y2)])
            
        else:
            slope = math.atan2((line[1][1] - line[0][1]), (line[1][0] - line[0][0]))
            #print("slope of line: " + str(slope))
            r = math.sqrt(1 + (slope * slope))                             
            x1 = int(line[0][0] - 10000 * math.cos(slope))
            y1 = int(line[0][1] - 10000 * math.sin(slope))

            x2 = int(line[1][0] + 10000 * math.cos(slope))
            y2 = int(line[1][1] + 10000 * math.sin(slope))
        
            extendedLines.append([(x1, y1), (x2, y2)])

    print("extended lines")
    print(extendedLines)
    return extendedLines

kernel = np.ones((3, 3), np.uint8)


# Estimates the are of plant cover within a segment
def EstimateSegmentArea(img, numSegments, filename):
    # Dictionary that contains the segment number and plant cover within the given image
    segmentAreas = defaultdict(lambda: "Not Present")
    # Height and width of the plant image
    gridHeight = img.shape[0]
    gridWidth = img.shape[1]
    
    # Computes the height of each grid segment
    gridSegmentHeight = img.shape[0]//numSegments
        
    x = 0
    y = 0
    count = 0
    
    # Estimates the area within each grid segment
    for i in range(numSegments):
        # Crop the image and extract the region corresponding to a single segment
        imgCrop = img[y:y+gridSegmentHeight, x:x+gridWidth]
        imgCrop[np.array(imgCrop) > 0] = 255
        
        # Compute the number of white pixels i.e. the pixels representing the plant 
        plantPixels = cv2.countNonZero(imgCrop)
        
        # Compute the number of pixels in the image
        totalPixels = imgCrop.shape[0] * imgCrop.shape[1]
        
        # Compute percentage of plant cover in the image
        plantPixelPercentage = plantPixels/totalPixels
                
        segmentAreas[filename.split("\\")[-1] + "_" + str(count)] = plantPixelPercentage
        
        cv2.imwrite("C:\\Users\\marven\\Documents\\Fall-2021\\Groundcover\\croppedImages\\crop_{0}_{1}.jpg".format(filename.split("\\")[-1], count), imgCrop)
        
        # Increment height to be the height of the next segment for cropping
        y = y + gridSegmentHeight

        count = count + 1
        
    return segmentAreas
        
# Finds the index of the largest contours within a given array of contours        
def FindLargestContourIndex(contours):
    maxArea = -1
    maxIndex = -1
    contindex_area = {}
    for i,v in enumerate(contours):
        contArea = cv2.contourArea(v)
        contindex_area[i] = contArea
        if(contArea > maxArea):
            maxIndex = i
            maxArea = contArea
            
    contindex_area = sorted(contindex_area.items(), key = lambda x: x[1], reverse = True)
    print(maxArea)
    print(maxIndex)
    print(contindex_area[0][0])
    return contindex_area[0][0], contindex_area[1][0]
        
def GroupLinesbyDirectionandFindValidLines(horizontalLines, VerticalLines, imgShape):
    center = (imgShape[0]//2, imgShape[1]//2)
        
    # Find the center of the image along the x axis
    # It helps determine if a line is to the left or right boundary of the PVC grid
    
    leftLines = []
    rightLines = []
    for line in lines:
        (x1, y1) = (line[0][0], line[0][1])
        (x2, y2) = (line[0][2], line[0][3])
        if x1 < center[1]:
            leftLines.append(line)
        else:
            rightLines.append(line)
            
    topLines = []
    bottomLines = []
    for line in lines:
        (x1, y1) = (line[0][0], line[0][1])
        (x2, y2) = (line[0][2], line[0][3])
        if y1 < center[0]:
            topLines.append(line)
        else:
            bottomLines.append(line)
            
    leftLines = sorted(leftLines, key = lambda x: x[0][0])
    rightLines = sorted(rightLines, key = lambda x: x[0][0])
    topLines = sorted(topLines, key = lambda x: x[0][1])
    bottomLines = sorted(bottomLines, key = lambda x: x[0][1])
    
    if len(leftLines) == 0:
        leftLines = [(0, 0), (0, imgShape[0])]
    
    if len(rightLines) == 0:
        rightLines = [(imgShape[1], 0), (imgShape[1], imgShape[0])]
        
    if len(topLines) == 0:
        topLines = [(0, 0), (imgShape[1], 0)]
        
    if len(bottomLines) == 0:    
        bottomLines = [(0, imgShape[0]), (imgShape[1], imgShape[0])]

    
#     print("leftLines")
#     print(leftLines)
    
#     print("rightLines")
#     print(rightLines)
    
#     print("topLines")
#     print(topLines)
    
#     print("bottomLines")
#     print(bottomLines)
    
    leftLine = leftLines[0]
    rightLine = rightLines[0]
    topLine = topLines[0]
    bottomLine = bottomLines[0]
    
    validLines = []
    validLines.append(leftLine)
    validLines.append(rightLine)
    validLines.append(topLine)
    validLines.append(bottomLine)
    
    return validLines
    
    
def RemoveGridfromSLICImage(slicImage):
    slicImage = cv2.cvtColor(slicImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(slicImage, 158, 255, cv2.THRESH_BINARY)
        
    # Create a kernel to operate on the grayscale image. 
    # The size of the kernel is a standard (3, 3) 
    kernel = np.ones((3, 3), np.uint8)        

    # Perform canny edge detection to detect the edges in the image
    edges = cv2.Canny(image=thresh, threshold1=100, threshold2=200) # Canny Edge Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, lines = 4, minLineLength = 10, maxLineGap = 500)
    
    horizontalLines, verticalLines = RetrieveParallelLines(lines)
    
    validLines = GroupLinesbyDirectionandFindValidLines(horizontalLines, verticalLines, slicImage.shape)

    formattedLines = []
    for line in validLines:
        formattedLines.append([(line[0][0], line[0][1]), (line[0][2], line[0][3])])
    
    validLines = ExtendLinesInfinitely(formattedLines, slicImage.shape[0])
    
    
    for line in validLines:
        thresh = cv2.line(thresh, line[0], line[1], (rng.randint(254, 255), rng.randint(250, 255), rng.randint(250, 255)), 2)
        
        
    #cv2.imwrite("gridRemoval.jpg", thresh)
    
    
if os.path.isfile("KuraFieldPicture_7.3.17.txt"):
        os.remove("KuraFieldPicture_7.3.17.txt")
outputFile = open("KuraFieldPicture_7.3.17.txt", "w")    
        
for filename in glob.glob(os.path.join(CURRENT_DIRECTORY, "OneSegmentImages", "*.jpg")):
    # read the image file
    img = cv2.imread(filename)        

    # create a copy and operate on it so as to leave the original image untouched
    img_copy = img.copy()
    
    slic_img_copy = img.copy()

    # convert from BGR to HSV color space
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

    # lower and upper bounds of H, S and V to identify the PVC grid in image 
    # These values are determined empirically and vary by the color of the PVC grid in consideration
    lowerRange = np.array([0, 0, 193])
    upperRange = np.array([180, 40, 255])        

    # Extracting all pixels in image that correspond to the PVC grid
    thresh = cv2.inRange(img_copy, lowerRange, upperRange)
    
    #cv2.imwrite('thresh2.jpg', thresh)
    
    # Binary threshold the image to set all PVC grid pixels to 255. 
    ret, gray = cv2.threshold(thresh, 220, 255, cv2.THRESH_BINARY) 
        
    img_gray = gray.copy()#cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # Create a mask image where all pixels are set to 0 (black image)
    ctr_img = np.zeros(img_gray.shape, np.uint8)

    # Create a kernel to operate on the grayscale image. 
    # The size of the kernel is a standard (3, 3) 
    kernel = np.ones((3, 3), np.uint8)

    # Apply dilation to enhance the PVC grid on the image so contours can be extracted
    img_gray = cv2.dilate(img_gray, kernel, iterations = 2)

    # Perform canny edge detection to detect the edges in the image
    edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200) # Canny Edge Detection
        
    #cv2.imwrite(os.path.join(CURRENT_DIRECTORY, "canny", filename.split("\\")[-1]), edges)    
    WriteImage(edges, "canny", filename)
    
    # Dilate the image on which edges have been detected
    # This step is optional but only performed to accentuate the edges for better contour detection. In a way, it is also unusual.
    #edges = cv2.dilate(edges, kernel, iterations = 1)
    
    found = False
    votes = 180
    while(not found and votes > 0):
        # Compute Probabilistic Hough Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, votes, lines = 4, minLineLength = 10, maxLineGap = 500)   
                    
        print(len(lines))
        if(lines is None):
            votes = votes - 10
            continue

        if(votes == 0):
            break                  

        if(len(lines) == 0):
            votes = votes - 10
            continue

        found = True
        
        # Retrieve parallel horizontal lines i.e. the portion of the grid along the horizontal
        horizontalLines, verticalLines = RetrieveParallelLines(lines)
                              
        
        # Merge the parallel lines together to reduce the number of lines required to be processed.
        merged_horizontal_lines_all = MergeLines(horizontalLines)
        
        merged_vertical_lines_all = MergeLines(verticalLines)
        
        validLines = GetVerticalandHorizontalLinesforGrid(merged_horizontal_lines_all, merged_vertical_lines_all, img_gray.shape[1], img_gray.shape[0])               
        
#         for line in validLines:
#             (x1, y1) = (line[0][0], line[0][1])
#             (x2, y2) = (line[1][0], line[1][1])                                                            
#             found = True
#             #validLines.append([(x1, 0), (x2, ctr_img.shape[0] - 20)])
#             cv2.line(img, (x1, y1), (x2, y2), (rng.randint(254, 255), rng.randint(250, 255), rng.randint(250, 255)), 2)
        
        pts = FindPointsofIntersectionofLines(validLines, img_gray.shape[1], img_gray.shape[0])
        
        unwarpedimg = UnwarpImage(pts, img)
                
        #cv2.imwrite(os.path.join(CURRENT_DIRECTORY, "OneSegmentUnwarped", filename.split("\\")[-1]), unwarpedimg)
        WriteImage(unwarpedimg, "OneSegmentUnwarped", filename)

# Step 1: Identify the PVC grid contour in the image
# # Read the image as a grayscale image
print(UNWARPED_IMAGE_PATH + "*.jpg")
for filename in glob.glob(UNWARPED_IMAGE_PATH + "*.jpg"):
    # read the image file
    img = cv2.imread(filename)  

    print("here")

    # create a copy and operate on it so as to leave the original image untouched
    img_copy = img.copy()
    
    slic_img_copy = img.copy()

    # convert from BGR to HSV color space
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

    # lower and upper bounds of H, S and V to identify the PVC grid in image 
    # These values are determined empirically and vary by the color of the PVC grid in consideration
    lowerRange = np.array([0, 0, 193])
    upperRange = np.array([180, 40, 255])        

    # Extracting all pixels in image that correspond to the PVC grid
    thresh = cv2.inRange(img_copy, lowerRange, upperRange)
    
    #cv2.imwrite('thresh2.jpg', thresh)
    
    # Binary threshold the image to set all PVC grid pixels to 255. 
    ret, gray = cv2.threshold(thresh, 220, 255, cv2.THRESH_BINARY) 
        
    img_gray = gray.copy()#cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # Create a mask image where all pixels are set to 0 (black image)
    ctr_img = np.zeros(img_gray.shape, np.uint8)

    # Create a kernel to operate on the grayscale image. 
    # The size of the kernel is a standard (3, 3) 
    kernel = np.ones((3, 3), np.uint8)

    # Apply dilation to enhance the PVC grid on the image so contours can be extracted
    img_gray = cv2.dilate(img_gray, kernel, iterations = 2)

    # Perform canny edge detection to detect the edges in the image
    edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200) # Canny Edge Detection
    
    #cv2.imwrite(os.path.join(CURRENT_DIRECTORY, "canny", filename.split("\\")[-1]), edges)    
    WriteImage(edges, "canny", filename)
    
    # Dilate the image on which edges have been detected
    # This step is optional but only performed to accentuate the edges for better contour detection. In a way, it is also unusual.
    #edges = cv2.dilate(edges, kernel, iterations = 1)
    
    found = False
    votes = 160
    while(not found and votes > 0):
        # Compute Probabilistic Hough Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, votes, lines = 4, minLineLength = 1, maxLineGap = 500)   
                    
        print(len(lines))
        if(lines is None):
            votes = votes - 10
            continue

        if(votes == 0):
            break                  

        if(len(lines) == 0):
            votes = votes - 10
            continue

        found = True
        
        # Retrieve parallel horizontal lines i.e. the portion of the grid along the horizontal
        horizontalLines, verticalLines = RetrieveParallelLines(lines)
                              
        
        # Merge the parallel lines together to reduce the number of lines required to be processed.
        merged_horizontal_lines_all = MergeLines(horizontalLines)
        
        merged_vertical_lines_all = MergeLines(verticalLines)
        
        validLines = GetVerticalandHorizontalLinesforGrid(merged_horizontal_lines_all, merged_vertical_lines_all, img_gray.shape[1], img_gray.shape[0])       
        
        
        for line in validLines:
            (x1, y1) = (line[0][0], line[0][1])
            (x2, y2) = (line[1][0], line[1][1])                                                            
            found = True
            #validLines.append([(x1, 0), (x2, ctr_img.shape[0] - 20)])
            cv2.line(img, (x1, y1), (x2, y2), (rng.randint(254, 255), rng.randint(250, 255), rng.randint(250, 255)), 2)
            
        cv2.imwrite("borderLines.jpg", img)

        print("original img shape: " + str(img.shape))  
        
        pts = FindPointsofIntersectionofLines(validLines, img.shape[1], img.shape[0])
        
        print("border pts")
        print(pts)
        pts[pts < 0] = 0
        print(pts)        

        
        cropped, polygonalMask = ExtractBitwiseImagesofPVCGrid(pts, img)
        
        t0 = time.time()

        labels1 = segmentation.slic(cropped, compactness=300, n_segments=1000, enforce_connectivity = False)

        t1 = time.time()

        print(t1 - t0)

        out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

        #cv2.imwrite(PLANT_OUTPUT_PATH + filename.split("\\")[-1], out1)
        WriteImage(out1, "boundary_lines_grid", filename)
        
        maskImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

        pvcFrameImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)

        labelIds = np.unique(labels1)

        centers_labels = {}

        for i in labelIds:
            centers_labels[i] = np.mean(np.nonzero(labels1==i),axis=1)

        plantLocations = []
        
        print("plant locations")
                
        plantLocations = list(map(lambda x: x[0], centers_labels.items()))

    #plantLocations = list(map(lambda x: x[0] if (np.argmax(out1[int(x[1][0]), int(x[1][1])]) == 1 and np.max(out1[int(x[1][0]), int(x[1][1])]) > 0) else None, centers_labels.items()))

        plantLocations = list(filter(partial(is_not, None), plantLocations))

    #pvcframeLocations = list(filter(partial(is_not, None), pvcframeLocations))

        for i in plantLocations:
            lab = np.where(labels1 == i)
            for val in range(len(lab[0])):
                cv2.circle(maskImg, (lab[1][val], lab[0][val]), radius = 0, color = (255, 255, 255), thickness = -1)

        #maskImg = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)

        bitwise = cv2.bitwise_and(slic_img_copy, slic_img_copy, mask = maskImg)

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

        cv2.imwrite(FINAL_HSV_OUTPUT_PATH + filename.split("\\")[-1], bitwise_and_img2)

        subImg = cv2.subtract(bitwise_and_img, bitwise_and_img2)

        #cv2.imwrite(SUBTRACTED_IMAGE_OUTPUT_PATH + filename.split("\\")[-1], subImg)
        WriteImage(subImg, "subtracted_output", filename)
        
        plantCover = EstimatePlantCover(polygonalMask, subImg)
        
        print("writing output " + str(plantCover))
        
        outputFile.write(filename + " " + str(plantCover) + "\n")
        
    


    
    
    
    
