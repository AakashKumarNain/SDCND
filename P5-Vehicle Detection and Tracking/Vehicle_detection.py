import os 
import cv2
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from natsort import natsorted
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label



# Define a function for computing color histogram
def color_hist(img, nbins=32, bins_range=(0,256), debug=False):
    # Compute histograms for each channel
    ch1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ch2_hist = np.histogram(img[:,:,1], bins=nbins, range= bins_range)
    ch3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Get bin edges
    bin_edges = ch1_hist[1]
    # Generate bin centers
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1]) / 2
    # Concatenate the histograms into a single feature vector
    feature_vector = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
    if debug == True:
        # Return individual histograms, bin centers and feature vector
        return ch1_hist, ch2_hist, ch3_hist, bin_centers, feature_vector
    return feature_vector      




# Define spatial binning of colors 
def bin_spatial(img, color_space='BGR', size=(32,32)):
    # Convert to the specified color space if other than RGB
    if color_space != 'BGR': 
        if color_space == 'HSV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if color_space == 'YCrCb':
            feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if color_space == 'HLS':
            feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)    
    else:
        feature_img = np.copy(img)
     
    feature_img = feature_img.astype(np.float32)/255
    #print(np.min(feature_img), np.max(feature_img))
        
    features = cv2.resize(feature_img, size).ravel()
    return features




# Define a function to compute and return HOG features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                                       cells_per_block=(cell_per_block, cell_per_block),
                                       transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, 
                            pixels_per_cell=(pix_per_cell, pix_per_cell),
                            cells_per_block=(cell_per_block, cell_per_block),
                            transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features

    


# Define a function for doing sliding window
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None,None], xy_window=(96,96), xy_overlap=(0.75,0.75)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    # Compute the span of the region to be searched 
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of sliding windows
    nx_windows = np.int(xspan / nx_pix_per_step)
    ny_windows = np.int(yspan / ny_pix_per_step)
    
    # Initialize an empty list for appending windows
    windows_list = []
    # Loop through the number of windows in x and y directions
    for y_window in range(ny_windows):
        for x_window in range(nx_windows):
            start_x = x_window*nx_pix_per_step + x_start_stop[0]
            end_x = start_x + xy_window[0]
            start_y = y_window*ny_pix_per_step + y_start_stop[0]
            end_y = start_y + xy_window[1]
            
            windows_list.append(((start_x, start_y), (end_x, end_y)))
    
    return windows_list




# Define a function for drawing boxes
def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    img_copy = np.copy(img)
    for box in bboxes:
        cv2.rectangle(img_copy, box[0], box[1], color, thick)
    return img_copy    




# Define a function to extract features from the images
def extract_features(img, color_space='BGR', spatial_size=(32,32), hist_bins=32, hist_range=(0,256),orient=9, 
                pix_per_cell=8, cell_per_block=2, hog_channel='ALL',spatial_feat=True, hist_feat=True, hog_feat=True):
    
    if img.shape != (64,64,3):
    	img = cv2.resize(img, (64,64))

    # Create a list to append features
    features = []
    cspace = color_space

    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if color_space == 'YCrCb':
            feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if color_space == 'HLS':
            feature_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)     
    else:
        feature_img = np.copy(img)
    
    feature_img = feature_img.astype(np.float32)/255 
    #print(np.min(feature_img), np.max(feature_img))
        
    # Compute spatial features if set to true
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_img, color_space=cspace, size=spatial_size)
        features.append(spatial_features)
    
    # Compute histogram features if set to true
    if color_hist == True:
        color_features = color_hist(feature_img, nbins=hist_bins, bins_range=hist_range)
        features.append(color_features)
    
    # Compute hog features if set to true
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features= []
            for i in range(feature_img.shape[2]):
                hog_features.extend(get_hog_features(feature_img[:,:,i], orient, pix_per_cell, cell_per_block, feature_vec=True ))
        else:
            hog_features = get_hog_features(feature_img[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=True)
        features.append(hog_features)
    
    # return a concatenated list of features
    return np.concatenate((features))       




# Function to search a window where the classifier reports a true prediction
def search_windows(img, windows, clf, scaler, color_space='BGR',spatial_size=(16, 16), hist_bins=16,
                   orient=9,pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #print(test_img.shape)
        # 4) Extract features
        features = extract_features(test_img,color_space=color_space, orient=orient, spatial_size=spatial_size, hist_bins=hist_bins,
                         					 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                         					spatial_feat=spatial_feat, hist_feat=hog_feat, hog_feat=hist_feat, hog_channel=hog_channel)

        #print(features.shape)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 5) Predict using your classifier
        prediction = clf.predict(test_features)
        # 6) If positive (prediction == 1) then save the window
        if clf.decision_function(test_features) >=0.65 and prediction ==1::
            on_windows.append(window)
    # 7) Return windows for positive detections
    return on_windows






# Define a function to produce heatmap
def add_heatmap(hmap, bboxes):
    for box in bboxes:
        hmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return hmap    



# Apply thresholding to the heatmap 
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



# Define a function to draw labeled boxes
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img




# Load the dataset containing images of vehicles and non-vehicles
def load_data():
    car_images = glob.glob('./Dataset/vehicles/**/*.png', recursive=True)
    nocar_images = glob.glob('./Dataset/non-vehicles/**/*.png', recursive=True)

    x_train = []
    x_valid = []
    y_train = []
    y_valid = []

    # Do train/validation split _properly_
    for target, dataset in enumerate([nocar_images, car_images]):
        folders = list(set(['/'.join(f.split('/')[:-1]) for f in dataset])) # Get unique folders
        print(folders)
        for folder in folders:
            # Sort each folder by time series
            folder_files = natsorted([f for f in dataset if folder in f])
            # Split into train and validation
            folder_files_train = folder_files[:int(len(folder_files) * 0.75)]
            folder_files_valid = folder_files[int(len(folder_files) * 0.75):]

            print('Loading ({}) {} - {} train {} valid'.format(target, folder, len(folder_files_train), 
                                                               len(folder_files_valid)))

            folder_imgs_train = [cv2.imread(f) for f in folder_files_train]
            folder_imgs_valid = [cv2.imread(f) for f in folder_files_valid]
            
           
            x_train.extend(folder_imgs_train)
            x_valid.extend(folder_imgs_valid)
            y_train.extend([target for _ in folder_files_train])
            y_valid.extend([target for _ in folder_files_valid])

    x_train = np.array(x_train)   
    y_train = np.array(y_train) 
    x_valid = np.array(x_valid) 
    y_valid = np.array(y_valid) 

    print("Shape of train data : ", x_train.shape, y_train.shape)
    print("Shape of validation data : ", x_valid.shape, y_valid.shape)
    
    return x_train, y_train, x_valid, y_valid



# Run a classifier 
def run_classifier(xtrain, ytrain, xvalid, yvalid):
    # Standardise the dataset
    scaler = StandardScaler()
    # Fit the scaler on the training data
    scaler = scaler.fit(xtrain)
    # Transform xtrain and xvalid datasets
    xtrain = scaler.transform(xtrain)
    xvalid = scaler.transform(xvalid)
    
    # Create an instance of the classisifer
    clf = LinearSVC(C=0.1, penalty='l2', random_state=111)
    #clf = RandomForestClassifier(max_depth=6, n_estimators=100,min_samples_split=2, oob_score=True, random_state=111)
    clf.fit(xtrain, ytrain)
    
    # Check training and validation accuracy
    ptrain = clf.predict(xtrain)
    pvalid = clf.predict(xvalid)
    print("Training accuracy : ", (ptrain == ytrain).mean())
    print("Validation accuracy : ", (pvalid == yvalid).mean())
    
    #return the classifier and the scaler
    return clf, scaler



# Get the training and validation sets
x_train, y_train, x_valid, y_valid = load_data()


# Extract features for train and validation sets
xtrain = np.array([extract_features(img,color_space='HSV',orient=9,cell_per_block=2,pix_per_cell=8,hist_bins=32,
            hist_range=(0,256),hist_feat=True,hog_channel='ALL',hog_feat=True,spatial_feat=True) for img in x_train])

xvalid = np.array([extract_features(img,color_space='HSV',orient=9,cell_per_block=2,pix_per_cell=8,hist_bins=32,
            hist_range=(0,256),hist_feat=True,hog_channel='ALL',hog_feat=True,spatial_feat=True) for img in x_valid])

print("Training data shape : ", xtrain.shape)
print("Validation shape : ", xvalid.shape)

#del x_train, x_valid



clf, scaler = run_classifier(xtrain, y_train, xvalid, y_valid)



# Check the pipeline on the test images
def process_test_images(path):
    image = cv2.imread(path)
    # print(image.shape)
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[350,600], xy_window=(96,96), 
                           xy_overlap=(0.75, 0.75))

    color_space='HSV'
    spatial_size = (32,32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    spatial_feat = True
    hist_feat = True
    hog_feat = True
    
    hot_windows = search_windows(image, windows, clf, scaler, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)
 
    heatmap = np.zeros_like(image[:,:,1]).astype(np.float)
    heatmap = add_heatmap(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, 1)
    heatmap = np.clip(heatmap, 0, 255)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 255, 0), thick=6)

    
    f,ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(window_img)
    ax[1].imshow(heatmap,cmap='hot')
    mimg.imsave( path[:-4]  + '_processed.jpg', window_img)
    mimg.imsave( path[:-4] + '_heatmap.jpg', heatmap, cmap='hot')
    plt.show()




# Get the test images and test the pipeline on the test images
test_images = glob.glob('./test_images/*.jpg')
for item in test_images:
    process_test_images(item)


# Define a class for queuing up the detected frames
class FrameQueue:
    def __init__(self, max_frames):
        self.frames = []
        self.max_frames = max_frames

    def enqueue(self, frame):
        self.frames.insert(0, frame)

    def _size(self):
        return len(self.frames)

    def _dequeue(self):
        num_element_before = len(self.frames)
        self.frames.pop()
        num_element_after = len(self.frames)

        assert num_element_before == (num_element_after + 1)

    def sum_frames(self):
        if self._size() > self.max_frames:
            self._dequeue()
        all_frames = np.array(self.frames)
        return np.sum(all_frames, axis=0)




# Define a class for vehicle detection
class VehicleDetector:
    def __init__(self, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat,
                 		hist_feat, hog_feat, y_start_stop, x_start_stop, xy_window, xy_overlap, heat_threshold, scaler, classifier):
        
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.y_start_stop = y_start_stop
        self.x_start_stop = x_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap
        self.heat_threshold = heat_threshold
        self.scaler = scaler
        self.classifier = classifier
        self.frame_queue = FrameQueue(5)


    def detect(self, input_image):
        img_copy = np.copy(input_image)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

        slided_windows = slide_window(img_copy, x_start_stop=self.x_start_stop,
                                      				y_start_stop=self.y_start_stop,
                                      				xy_window=self.xy_window, xy_overlap=self.xy_overlap)

        on_windows = search_windows(img_copy, slided_windows, self.classifier, self.scaler,
                                    					color_space=self.color_space, spatial_size=self.spatial_size,
                                    					hist_bins=self.hist_bins, orient=self.orient,
                                    					pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                    					hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                    					hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        heatmap = np.zeros_like(img_copy)
        heatmap = add_heatmap(heatmap, on_windows)
        self.frame_queue.enqueue(heatmap)
        all_frames = self.frame_queue.sum_frames()
        heatmap = apply_threshold(all_frames, self.heat_threshold)
        
        labels = label(heatmap)

        image_with_bb = draw_labeled_bboxes(input_image, labels)
        return image_with_bb


# Define the parameters for the pipeline
color_space='HSV'
spatial_size = (32,32)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True
x_start_stop = [None, None]
y_start_stop = [350,600]
xy_window = (96,96)
xy_overlap = (0.75,0.75)
heat_threshold = 1

# Instantiate the vehicle detector class
vehicle_detector = VehicleDetector(color_space=color_space,
                                   orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel,
                                   spatial_size=spatial_size,
                                   hist_bins=hist_bins,
                                   spatial_feat=spatial_feat,
                                   hist_feat=hist_feat,
                                   hog_feat=hog_feat,
                                   y_start_stop=y_start_stop,
                                   x_start_stop=x_start_stop,
                                   xy_window=xy_window,
                                   xy_overlap=xy_overlap,
                                   heat_threshold = heat_threshold,
                                   scaler=scaler,
                                   classifier=clf)

# Specify the input and output file for test video
input_file = './test_video.mp4'
output_file = './processed_test_video.mp4'


# run the pipeline on the video and check the output
clip = VideoFileClip(input_file)
out_clip = clip.fl_image(vehicle_detector.detect)
out_clip.write_videofile(output_file, audio=False)


# Specify the input and output file for test video
input_file = './project_video.mp4'
output_file = './processed_project_video_3.mp4'

# run the pipeline on the video and check the output
clip = VideoFileClip(input_file)
out_clip = clip.fl_image(vehicle_detector.detect)
out_clip.write_videofile(output_file, audio=False)

