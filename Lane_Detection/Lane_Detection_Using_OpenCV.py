#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#Function to dislay image needed
def Show_Image(img):
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # Detecting the Line patches 

# In[ ]:


#Converting Image to Gray Scale
image = cv2.imread('./Road_With_Markings.jpeg')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)


#Defining the Highlight colors which are White and Yellow

lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
upper_yellow = np.array([30, 255, 255], dtype = 'uint8') 

mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(image_gray, 200, 255)

patch = cv2.bitwise_or(mask_yellow,mask_white)

patch_image = cv2.bitwise_and(image_gray,patch)

imshape = patch_image.shape
vertices = np.array([[(int(0.21*imshape[1]),imshape[0]),(int(0.44*imshape[1]), int(0.59*imshape[0])), (int(0.50*imshape[1]), int(0.59*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)


# # Clearing the Noise and applying the Edge detection
# # Detecting the lines in the image

# ## Necessary Functions

# In[ ]:


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255# white
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# In[ ]:


kernel_size = 5
Gauss_img = cv2.GaussianBlur(patch_image, (kernel_size, kernel_size), 0)
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(Gauss_img, low_threshold, high_threshold)
imshape = canny_edges.shape
vertices = np.array([[(int(0.21*imshape[1]),imshape[0]),(int(0.44*imshape[1]), int(0.59*imshape[0])), (int(0.50*imshape[1]), int(0.59*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
img_with_Roi = region_of_interest(canny_edges,vertices)
lines = cv2.HoughLinesP( img_with_Roi, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]), minLineLength=60,
    maxLineGap=30)


# # Detecting and marking the Lines

# ## Necessary Functions

# In[ ]:


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # Function has been written to work with Challenge video as well
    # b -0, g-1, r-2 
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # At the bottom of the image, imshape[0] and top has been defined as 330
    imshape = img.shape 
    
    slope_left=0
    slope_right=0
    leftx=0
    lefty=0
    rightx=0
    righty=0
    i=0
    j=0
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope >0.1: #Left lane and not a straight line
                # Add all values of slope and average position of a line
                slope_left += slope 
                leftx += (x1+x2)/2
                lefty += (y1+y2)/2
                i+= 1
            elif slope < -0.2: # Right lane and not a straight line
                # Add all values of slope and average position of a line
                slope_right += slope
                rightx += (x1+x2)/2
                righty += (y1+y2)/2
                j+= 1
    # Left lane - Average across all slope and intercepts
    if i>0: # If left lane is detected
        avg_slope_left = slope_left/i
        avg_leftx = leftx/i
        avg_lefty = lefty/i
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_l = int(((int(0.97*imshape[0])-avg_lefty)/avg_slope_left) + avg_leftx)
        xt_l = int(((int(0.61*imshape[0])-avg_lefty)/avg_slope_left)+ avg_leftx)

    else: # If Left lane is not detected - best guess positions of bottom x and top x
        xb_l = int(0.21*imshape[1])
        xt_l = int(0.43*imshape[1])
    
    # Draw a line
    cv2.line(img, (xt_l, int(0.61*imshape[0])), (xb_l, int(0.97*imshape[0])), color, thickness)
    
    #Right lane - Average across all slope and intercepts
    if j>0: # If right lane is detected
        avg_slope_right = slope_right/j
        avg_rightx = rightx/j
        avg_righty = righty/j
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_r = int(((int(0.97*imshape[0])-avg_righty)/avg_slope_right) + avg_rightx)
        xt_r = int(((int(0.61*imshape[0])-avg_righty)/avg_slope_right)+ avg_rightx)
    
    else: # If right lane is not detected - best guess positions of bottom x and top x
        xb_r = int(0.89*imshape[1])
        xt_r = int(0.53*imshape[1])
    
    # Draw a line    
    cv2.line(img, (xt_r, int(0.61*imshape[0])), (xb_r, int(0.97*imshape[0])), color, thickness)

    
def lane_detector(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #print(image.shape)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(patch_image, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 10
    high_threshold = 150
    edges = cv2.Canny(Gauss_img, low_threshold, high_threshold)

    # Create masked edges image
    imshape = image.shape
    vertices = np.array([[(int(0.21*imshape[1]),imshape[0]),(int(0.44*imshape[1]), int(0.59*imshape[0])), (int(0.50*imshape[1]), int(0.59*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)


    # Define the Hough transform parameters and detect lines using it
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 60 #minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments

    line_img = cv2.HoughLinesP( img_with_Roi, rho=1, theta=np.pi / 180, threshold=10, lines=np.array([]), minLineLength=30,
    maxLineGap=25)
    
    draw_lines(image, line_img, color=[0, 0, 255], thickness=3)

    return image


# In[ ]:


#create VideoCapture object and read from video file
cap = cv2.VideoCapture('solidWhiteRight.mp4')

#read until video is completed
while True:
    #capture frame by frame
    ret, frame = cap.read()
    final_img = lane_detector(frame)
    cv2.imshow('video', frame)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()
    


# In[ ]:





# In[ ]:




