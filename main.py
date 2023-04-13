"""
    Programmer: Emmanuel Samuel
    Date Created: 12th April 2023
    Date Revised: 13th April 2023
    Purpose: Social Distancing monitoring system using computer vision
"""

# import modules
import numpy as np
import cv2
import sys

# color of text defined
text_color = (24, 201, 255)
# tracker color defined
tracker_color = (255, 128, 0)
# warning color defined
warning_color = (24, 201, 255)
# font type defined
font_type = cv2.FONT_HERSHEY_SIMPLEX
# source of video defined
input_video = "people_library.mp4"

# define a list to hold background subtractor algorithms
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
# select an algorithm
bgs_type = BGS_TYPES[1]


# function to choose the type of kernel for image preprocessing
def get_kernel(kernel_type):
    if kernel_type == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if kernel_type == "opening":
        kernel = np.ones((3, 5), np.uint8)
    if kernel_type == "closing":
        kernel = np.ones((11, 11), np.uint8)

    return kernel


# function to apply morphological operations
def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, get_kernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, get_kernel("dilation"), iterations=2)

        return dilation


# function to implement the algorithm
def get_bgsubtractor(bgs_type):
    if bgs_type == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if bgs_type == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if bgs_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=80)
    if bgs_type == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if bgs_type == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Invalid detector")
    sys.exit(1)


# capture the video
cap = cv2.VideoCapture(input_video)
# apply the background subtractor function
bg_subtractor = get_bgsubtractor(bgs_type)
# Minimum Area defined
minArea = 1000
# Maximum Area defined
maxArea = 3000


# main function definition
def main():
    while cap.isOpened:
        # check for frames
        success, frame = cap.read()

        if not success:
            print("Finished processing the video")
            break

        # resize frame of video for processing to be faster
        frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        # apply the algorithm to frame
        bg_mask = bg_subtractor.apply(frame)
        # apply the morphological process
        bg_mask = get_filter(bg_mask, 'combine')
        # apply the median blur preprocessing specifying the GaussianBlur as 5
        bg_mask = cv2.medianBlur(bg_mask, 5)

        # to detect contour in the video
        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # iterate over contours and apply rectangle
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= minArea:
                """
                more functionalities can be added her such as to send an alarm
                """
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(frame, cnt, 1, tracker_color, 10)
                cv2.drawContours(frame, cnt, 1, (255, 255, 255), 1)

                if area >= maxArea:
                    cv2.rectangle(frame, (x, y), (x + 120, y - 13), (49, 49, 49), -1)
                    cv2.putText(frame, 'Warning', (x, y - 2), font_type, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.drawContours(frame, [cnt], -1, warning_color, 2)
                    cv2.drawContours(frame, [cnt], -1, warning_color, 1)

        # improve the img processing result using a bitwise function
        res = cv2.bitwise_and(frame, frame, mask=bg_mask)

        cv2.putText(res, bgs_type, (10, 50), font_type, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(res, bgs_type, (10, 50), font_type, 1, text_color, 2, cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', res)

        # press the q button to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


main()
