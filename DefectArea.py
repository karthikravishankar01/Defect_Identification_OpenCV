import cv2 as cv
import numpy as np


def defectArea(imgInput):
    canny = cv.Canny(imgInput, 200, 175)

    # Finding Contours
    contours, hierarchy = cv.findContours(
        canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Contour Approximation
    for c in contours:
        epsilon = 0.001*cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        defect_area = cv.drawContours(imgInput, [approx], 0, (36, 36, 216), 2)

    # extracting only white pixels
    fault_area_pixels = np.sum(imgInput == 255)

    cv.imshow('Defect Area', defect_area)
    cv.waitKey(0)
    return fault_area_pixels


imgInput = cv.imread('br8.png')
cv.imshow('Input Image', imgInput)
defect_output = defectArea(imgInput)
print("Defect Area: ", int(defect_output), "Pixels")
