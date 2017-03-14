import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import cameraCalibration as cC
from ad_Lanes import Lanes
from moviepy.editor import VideoFileClip
#get_ipython().magic('matplotlib inline')


# Define the path for images to be used for calibration
images_path = './camera_cal/*.jpg'
# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cC.calibrate_camera(file_path=images_path)
# Instantiate the Lanes class
lanes = Lanes()


# Define the pipeline
def pipeline_process(img):
    # Apply undistortion
    undist_img = cC.undistort_image(img, matrix=mtx, dist_coeff=dist)    
    # Get binary thresholded image
    binary_img, color_binary = cC.combined_binary_threshold(undist_img)  
    # Apply perspective transform
    pers_img, M, Minv = cC.perspective_transform(binary_img) 
    # Locate the lanes
    lanes.locate_lanes(pers_img)
    # Fit the lanes in the image
    lanes.fit_lanes()
    # Draw lanes on the image
    final_image = lanes.draw_lanes(undist_img, pers_img, Minv)
    return final_image




# Run the pipeline on the test images
images = glob.glob('./test_images/*.jpg')
f,ax = plt.subplots(2,4, figsize=(12,12))
for i,fname in enumerate(images):
    lanes = Lanes()
    img = mimg.imread(fname)
    img = cv2.GaussianBlur(img, (5,5), 0)
    final_image = pipeline_process(img)
    ax[i//4, i%4].imshow(final_image);
    ax[i//4, i%4].axis('off')


# Now run the pipeline on each frame of the project video
clip = VideoFileClip("./project_video.mp4")
output_video = "./project_video_processed_new.mp4"
output_clip = clip.fl_image(pipeline_process)
output_clip.write_videofile(output_video, audio=False)





