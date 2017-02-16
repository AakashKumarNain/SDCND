import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Lanes:
    def __init__(self, debug_mode=False):
        # Frame counter (used for finding new lanes)
        self.frame_number = 0
        # Lane locations from previous frame
        self.last_left_x = 0
        self.last_right_x = 0
        # Lane locations from current frame
        self.left_x = 0
        self.right_x = 0
        # Lane persp image
        self.left_lane_img = 0
        self.right_lane_img = 0
        # Lane persp debug image
        self.lane_debug_img = 0
        # Frames since lane last detected
        self.left_last_seen = 999
        self.right_last_seen = 999
        # Lane fit coords
        self.left_fitx = 0
        self.left_fity = 0
        self.right_fitx = 0
        self.right_fity = 0
        # Lane radius of curvature
        self.left_curverad = 0
        self.right_curverad = 0
        # Lanes found in last frame?
        self.left_lane_found  = False
        self.right_lane_found = False
        # Lane polynomial fits
        self.left_fit = []
        self.right_fit = []
        # Debug Mode
        self.debug_mode = debug_mode
        

    
    def search(self, hist):
        num_pixels_x = len(hist)
        # Separate left side and find the peak
        left_side = hist[0:int(num_pixels_x/2)]
        left_peak_x = np.argmax(left_side)
        # Separate right side and find the peak
        right_side = hist[int(num_pixels_x/2):]
        right_peak_x = np.argmax(right_side)
        # Adjustment
        right_offset = int(num_pixels_x/2)
        right_peak_x += right_offset
        return left_peak_x, right_peak_x

    
    # Get the x coordinates for the peaks 
    def get_two_peak_x_coords(self, hist, prev_left_x=-1, prev_right_x=-1, start_y=0, end_y=0, found_last_left=False, found_last_right=False, left_trend=0, right_trend=0):
        # Ge the total number of x pixels in the histogram
        num_pixels_x = len(hist)
        # Size of window 
        left_window = 40 
        right_window = 40

        found_left = True
        found_right = True

        if not found_last_left:     # Left lane was not detected in the last frame
            left_window = 60
        if not found_last_right:    # Right lane wasn't detected in the last frame
            right_window = 60
        if start_y == 720:
            left_window = 100 
            right_window = 100 

        left_offset = 0                
        
        if self.left_lane_found:
            # If there is a left lane, find the peak for it
            new_left_peak = int(self.left_fit[0]*start_y**2 + self.left_fit[1]*start_y + self.left_fit[2])
        else:
            # Else check for the values w.r.t. the last frame
            left_side = hist[prev_left_x + left_trend-left_window : prev_left_x + left_trend + left_window]
            new_left_peak = 0
            
            if len(left_side) > 0:
                new_left_peak = np.argmax(left_side)
                left_offset = prev_left_x + left_trend - left_window
            
            if new_left_peak == 0 or len(left_side) == 0:
                new_left_peak = prev_left_x + left_trend
                left_offset = 0
                found_left = False
        left_peak_x = new_left_peak + left_offset
        
        right_offset = 0
        if self.right_lane_found:
            # If the right lane is found, get the peak for it
            new_right_peak = int(self.right_fit[0]*start_y**2 + self.right_fit[1]*start_y + self.right_fit[2])
        else:
            # Else use the previous frame info to generate
            right_side = hist[prev_right_x + right_trend - right_window : prev_right_x + right_trend + right_window]
            new_right_peak = 0
            
            if len(right_side) > 0:
                new_right_peak = np.argmax(right_side)
                right_offset = prev_right_x + right_trend - right_window
            
            if new_right_peak == 0 or len(right_side) == 0:
                new_right_peak = prev_right_x + right_trend
                right_offset = 0
                found_right = False
        
        right_peak_x = new_right_peak + right_offset
        
        if start_y == 720:       # If we have just started to analyze the frames
            new_left_trend = 0
            new_right_trend = 0
        else:
            new_left_trend = left_peak_x - prev_left_x 
            new_right_trend = right_peak_x - prev_right_x

        return left_peak_x, right_peak_x, found_left, found_right, new_left_trend, new_right_trend
    
    
    
    # Locate the lanes in the image 
    def locate_lanes(self, img):
        # Check if this the first frame of video
        if self.frame_number == 0 or self.left_last_seen > 5 or self.right_last_seen > 5:
            # Generate histogram over bottom half of image
            histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
            # Find X coords of two peaks of histogram
            est_left_x, est_right_x = self.search(histogram)
            
        else:
            est_left_x = self.last_left_x
            est_right_x = self.last_right_x

        fallback_left_x = est_left_x
        fallback_right_x = est_right_x
        left_trend = 0
        right_trend = 0
    
        
        self.left_lane_img = np.zeros(img.shape[0:2], np.float32)
        self.right_lane_img = np.zeros(img.shape[0:2], np.float32)
        temp_img = img*255
        self.lane_debug_img = np.dstack((temp_img, temp_img, temp_img))
    
        found_last_left = False
        found_last_right = False
        left_window = 40
        right_window = 40
        left_conc_sections = 0
        right_conc_sections = 0

        # Run a sliding window up the image to detect pixels
        # There are a total of 9 windows used here 
        for i in range(10, 0, -1):
            start_y = int(i * img.shape[0]/10)
            end_y = int((i-1) * img.shape[0]/10)
            # Select the section of image accordingly
            img_sect = img[end_y:start_y,:]
            # Get the histogram
            histogram = np.sum(img_sect, axis=0)
            # Find values of x on the left side and right side along with other variables
            left_x, right_x, found_last_left, found_last_right, new_left_trend, new_right_trend = self.get_two_peak_x_coords(histogram, est_left_x, est_right_x, start_y, end_y, found_last_left, found_last_right, left_trend, right_trend)
            # Change the values of left_trend and right_trend based on new values
            left_trend = int((new_left_trend + left_trend) / 2)
            right_trend = int((new_right_trend + right_trend) / 2)
            # Store the left/right x values for bottom of image
            if i == 10:
                # Set the new last values
                self.left_x = left_x
                self.right_x = right_x

            if not found_last_left:
                left_x = fallback_left_x
                left_conc_sections = 0
            elif left_conc_sections > 1:
                fallback_left_x = left_x
            
            if not found_last_right:
                right_x = fallback_right_x
                right_conc_sections = 0
            elif right_conc_sections > 1:
                fallback_right_x = right_x

            if found_last_left:
                left_conc_sections += 1
            if found_last_right:
                right_conc_sections += 1
          
    
            # Fill in the left lane image
            left_mask = np.zeros_like(img_sect)
            left_mask[:,left_x-left_window:left_x+left_window]=1
            mask = (left_mask==1)
            self.left_lane_img[end_y:start_y,:] = img_sect & mask
    
            # Fill in the right lane image
            right_mask = np.zeros_like(img_sect)
            right_mask[:,right_x-right_window:right_x+right_window]=1
            mask = (right_mask==1)
            self.right_lane_img[end_y:start_y,:] = img_sect & mask
    
            # Set the new last values
            est_left_x = left_x
            est_right_x = right_x
        # Increase the number of frame        
        self.frame_number += 1

    def fit_lane(self, img):
        # Extract the list of x and y coords that are non-zero pixels
        xycoords = np.nonzero(img)
        x_arr = xycoords[1]
        y_arr = xycoords[0]
    
        # Fit a second order polynomial to each fake lane line
        fit = np.polyfit(y_arr, x_arr, deg=2)
        fitx = fit[0]*y_arr**2 + fit[1]*y_arr + fit[2]
    
        
        fitx = np.insert(fitx, 0, fit[0]*0**2 + fit[1]*0 + fit[2])
        fity = np.insert(y_arr, 0, 0)
        fitx = np.append(fitx, fit[0]*719**2 + fit[1]*719 + fit[2])
        fity = np.append(fity, 719)
    
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        y_eval = np.max(y_arr)
        fit_cr = np.polyfit(y_arr*ym_per_pix, x_arr*xm_per_pix, 2)
        fitx_cr = fit_cr[0]*(y_arr*ym_per_pix)**2 + fit_cr[1]*y_arr*ym_per_pix + fit_cr[2]
    
        # Get radius of curvature
        roc = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) /np.absolute(2*fit_cr[0])
        return fit, fitx, fity, roc



    def check_lane(self, new_roc, prev_roc, new_x, prev_x):
        # Check RoC against standards
        #if new_roc < 587 or new_roc > 4575:
        if new_roc < 587:
            return False

        # Check previous x coord versus current for major difference 
        delta = 15
        if new_x > prev_x + delta or new_x < prev_x - delta:
            return False

        # Check RoC against previous value
        max_roc = prev_roc * 100.0
        min_roc = prev_roc / 100.0
        if new_roc >= min_roc and new_roc <= max_roc:
            return True
        else:
            return False
    

    def fit_lanes(self):
        self.left_lane_found = False
        self.right_lane_found = False
        # Get new lane fit for left lane
        left_fit, left_fitx, left_fity, left_curverad = self.fit_lane(self.left_lane_img)
        # Only use this new lane fit if it's close to the previous one (for smoothing)
        if self.frame_number == 1 or self.check_lane(left_curverad, self.left_curverad, self.left_x, self.last_left_x):
            self.left_fit = left_fit
            self.left_fitx = left_fitx
            self.left_fity = left_fity
            self.left_curverad = left_curverad
            self.left_lane_found = True
            self.left_last_seen = 0
            self.last_left_x = self.left_x
        else:
            self.left_last_seen += 1
            

        # Get new lane fit for right lane
        right_fit, right_fitx, right_fity, right_curverad = self.fit_lane(self.right_lane_img)
        # Only use this new lane fit if it's close to the previous one (for smoothing)
        if self.frame_number == 1 or self.check_lane(right_curverad, self.right_curverad, self.right_x, self.last_right_x):
            self.right_fit = right_fit
            self.right_fitx = right_fitx
            self.right_fity = right_fity
            self.right_curverad = right_curverad
            self.right_lane_found = True
            self.right_last_seen = 0
            self.last_right_x = self.right_x
        else:
            self.right_last_seen += 1
        




    def draw_lanes(self, img, warped, Minv):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.left_fity]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.right_fity])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Draw lane lines only if lane was detected this frame
        if self.left_lane_found == True:
            cv2.polylines(color_warp, np.int_([pts_left]), False, (0,0,255), thickness=20)
        if self.right_lane_found == True:
            cv2.polylines(color_warp, np.int_([pts_right]), False, (255,0,0), thickness=20)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        weighted_img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        # Write the radius of curvature for each lane 
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_roc = "ROC: {0:.2f}m".format(self.left_curverad) 
        cv2.putText(weighted_img, left_roc, (10,630), font, 1, (255,255,255), 2)
        right_roc = "ROC: {0:.2f}m".format(self.right_curverad) 
        cv2.putText(weighted_img, right_roc, (1020,630), font, 1, (255,255,255), 2)
    
        # Write the x coords for each lane 
        left_coord = "X  : {0:.2f}".format(self.left_x) 
        cv2.putText(weighted_img, left_coord, (10,680), font, 1, (255,255,255), 2)
        right_coord = "X  : {0:.2f}".format(self.last_right_x) 
        cv2.putText(weighted_img, right_coord, (1020,680), font, 1, (255,255,255), 2)

        #Write dist from center
        perfect_center = 1280/2.
        lane_x = self.last_right_x - self.left_x
        center_x = (lane_x / 2.0) + self.left_x
        cms_per_pixel = 370.0 / lane_x   # US regulation lane width = 3.7m
        dist_from_center = (center_x - perfect_center) * cms_per_pixel + 28.0
        dist_text = "Dist from Center: {0:.2f} cms".format(dist_from_center)
        cv2.putText(weighted_img, dist_text, (450,50), font, 1, (255,255,255), 2)
    
        return weighted_img