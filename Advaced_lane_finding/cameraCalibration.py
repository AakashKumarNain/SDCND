import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg


# Define camera calibration 
def calibrate_camera(file_path):
    objpoints = []            ##3D points in real world space
    imgpoints = []            ##Correspoind 2D points in image plane

    # Prepare objects points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Get the image/s
    images = glob.glob(file_path)

    # Iterate over the files and apply processing steps
    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray,(9,6), None)

        # If the corners are found, add the values to imgpoints and objpoints
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and (optional) dispaly the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #plt.imshow(img)
            #plt.show()
        else:
            continue
    # Calibrate the camera        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs




# Define undistortion procedure
def undistort_image(img, matrix, dist_coeff):
    undistorted_image = cv2.undistort(img, matrix, dist_coeff, None, matrix)
    #plt.imshow(undistorted_image)
    return undistorted_image




# Define perspective transform
def perspective_transform(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Define source and destination co-ordinates
    src_cord = np.float32(([[240,719],[579,450],[712,450],[1165,719]]))
    dst_cord = np.float32([[300,719], [300,0], [900,0], [900,719]])
    
    # Find the perspective transformation matrix and inverse transformation matrix
    M = cv2.getPerspectiveTransform(src_cord, dst_cord)
    Minv = cv2.getPerspectiveTransform(dst_cord, src_cord)
    
    img_size = (img.shape[1], img.shape[0])
    
    # Warp the image
    warped_image = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_image, M, Minv





# Absolute Sobel threshold
def abs_sobel_thresh(img, orient='x', sobel_kernel =5 , thresh=(25,125)):
    # Convert the image to HSV
    image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = image[:,:,2]
    
    # Apply sobel 
    if orient == 'x':
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1,0, sobel_kernel)
    else:
        gradient = cv2.Sobel(gray, cv2.CV_64F, 0,1, sobel_kernel)
   
    # Take absolute
    abs_gradient = np.absolute(gradient)
    # Scale the image
    scaled_img = np.uint8(255*abs_gradient/np.max(abs_gradient))
    # Define a mask
    binary_output = np.zeros_like(scaled_img)
    binary_output[(scaled_img >= thresh[0]) & (scaled_img <= thresh[1])] =1
    return binary_output





# Color Thresholding
def color_threshold(img, color_thresh=(200,255)):
    # Convert to HSL space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Separate the s channel
    v_channel = hsv[:,:,2]
    r_channel = img[:,:,0]
    
    # Create and return a binary image
    v_binary = np.zeros_like(v_channel)
    r_binary = np.zeros_like(r_channel)
    v_binary[(v_channel >= color_thresh[0]) & (v_channel <= color_thresh[1])] = 1
    r_binary[(r_channel >= color_thresh[0]) & (r_channel <= color_thresh[1])] = 1
    
    # Combine thresholding of v and r channels
    comb_binary =  cv2.bitwise_or(v_binary, r_binary)
    return comb_binary






# Define combined thresholding for color, gradient,etc...
def combined_binary_threshold(img):
    # Find gradient thresholds
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=7, thresh=(50,150))
   
    # Combine all the gradient thresholds
    gmd_img = np.zeros_like(gradx)
    gmd_img[(gradx == 1)] =1
    
    # Find color threshold
    color_thresh_img = color_threshold(img)
    
    # Stack the color and gradient thresholds
    color_binary = np.dstack((np.zeros_like(gmd_img), gmd_img, color_thresh_img))
    
    # Combine the color and gradient thresholds
    combined_binary = np.zeros_like(gmd_img)
    combined_binary[(gmd_img ==1) | (color_thresh_img == 1)] = 1
    return combined_binary,  color_binary




