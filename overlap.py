from flask import Flask, Response
import cv2
import numpy as np
import time


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:37:26 2023

@author: samuel
"""

app = Flask(__name__)

# Initialize capture from each camera
cap1 = cv2.VideoCapture(30)  # Camera 1
cap2 = cv2.VideoCapture(0)   # Camera 2


#cap3 = cv2.VideoCapture(4)  # Camera 3
# cap1 = cap3
# cap2 = cap3

# cap2 = cv2.VideoCapture(2)  # Camera 2
# time.sleep(.5)
# result, image1 = cap1.read()

# image1 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)

# # # Displaying the image
# cv2.imshow("", image1)
# cv2.waitKey(0)

### ALGORITHM FOR CALIBRATION:
    
# Run ORB for 1000 frames between cap1 and cap 2
# Get Average M = M1
# Plug M1 into align_frames_m1

# Run ORB for 1000 frames between cap 3 and align_frames_m1
# Get average M = M2
# Plug M2 into align_frames_m2

# Run normally using  align_frames_m1 and align_frames_m2   

def get_avg_M(M_list):
    
    e1 =0 
    e2 =0
    e3 =0
    e4 =0
    e5 =0
    e6 =0
    e7 =0
    e8 =0
    e9 =0
    
    # Getting the average of M_list
    for element in M_list: 
        e1 = e1 + element[0,0]
        e2 = e2 + element[0,1]
        e3 = e3 + element[0,2]
        e4 = e4 + element[1,0]
        e5 = e5 + element[1,1]
        e6 = e6 + element[1,2]
        e7 = e7 + element[2,0]
        e8 = e8 + element[2,1]
        e9 = e9 + element[2,2]
        
    avg_m = [[e1/len(M_list),e2/len(M_list),e3/len(M_list)], [e4/len(M_list),e5/len(M_list),e6/len(M_list)], [e7/len(M_list),e8/len(M_list),e9/len(M_list)]]

    return avg_m



# Set the common resolution for all cameras
common_width = 640  
common_height = 480


def align_frames_sift(frame1, frame2):
    # Detect keypoints and descriptors (SIFT for example)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Get matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp frame1 to match frame2
    aligned_frame = cv2.warpPerspective(frame1, M, (frame2.shape[1], frame2.shape[0]))
    
    return aligned_frame


def align_frames_orb(frame1, frame2):
    
    ## SAMUEL: FASTER THAN SIFT, SIMILAR RESULTS
    
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(frame1,None)
    kp2, des2 = orb.detectAndCompute(frame2,None)
  
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Match descriptors.
    good_matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    #good_matches = sorted(matches, key = lambda x:x.distance)

    #good_matches = good_matches[:20]  

    # Get matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
                
        # Calculate homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        
        # Warp frame1 to match frame2
        aligned_frame = cv2.warpPerspective(frame1, M, (frame2.shape[1], frame2.shape[0]))
        
        #flat_M = M.flatten()
        #df.append(flat_M)
        #M_list.append(M)
        
        return aligned_frame
    
    except:
        return frame1
    
    
def calibrate_M():
    
    M_list = [] 
    i= 0
    
    while i<100: 
        
        i = i+1
        
        print("Calibrating... " + str(i) + " percent\n")
        
        # Capture synchronized frames from each camera
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
        
        frame1 = cv2.resize(frame1, (common_width, common_height))
        frame2 = cv2.resize(frame2, (common_width, common_height))
        
        # if i == 2:
        #     cv2.imwrite("thermal.jpg", frame1)
        #     cv2.imwrite("vga.jpg", frame2)
       
        
        # Convert to graycsale
        #frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        #img_blur1 = cv2.GaussianBlur(img1_gray, (3,3), 0)
        # Canny Edge Detection
        # frame1 = cv2.Canny(image=img_blur1, threshold1=100, threshold2=200) # Canny Edge Detection
        
        
        # Convert to graycsale
        #frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        #img_blur2 = cv2.GaussianBlur(img2_gray, (3,3), 0)
        #frame2 = cv2.Canny(image=img_blur2, threshold1=100, threshold2=200) # Canny Edge Detection
              
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(frame1,None)
        kp2, des2 = orb.detectAndCompute(frame2,None)
      
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Match descriptors.
        good_matches = bf.match(des1,des2)
    
        # Sort them in the order of their distance.
        #good_matches = sorted(matches, key = lambda x:x.distance)
    
        #good_matches = good_matches[:20]  
    
        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Close objects
        #src_pts_m = np.array([150,242,269,243,148,365,270,369]).reshape(-1,1,2)
        #dst_pts_m = np.array([189,154,380,153,189,269,377,275]).reshape(-1,1,2)
    
        # Far objects
        src_pts_m = np.array([88,143,476,156,68,252,481,266]).reshape(-1,1,2)
        dst_pts_m = np.array([180,157,412,172,164,275,414,289]).reshape(-1,1,2)
    
        # Calculate homography
        M, _ = cv2.findHomography(src_pts_m, dst_pts_m, cv2.RANSAC, 5.0)
        
       # print(M)
        
        # Warp frame1 to match frame2
        #aligned_frame = cv2.warpPerspective(frame1, M, (frame2.shape[1], frame2.shape[0]))
        
        #flat_M = M.flatten()
        #df.append(flat_M)
        
        if M is not None:
            M_list.append(M)
            
    
    avg_M = get_avg_M(M_list)
    
    avg_M = np.array(avg_M)
    
    return avg_M


def generate_frames():
    
    
    #M = calibrate_M()
    #print(M)
    # close_matrix:
    # M = np.array([[ 6.64835073e-01, -6.63578277e-02,  3.22757289e+01],
    #        [ 5.66972632e-02,  9.31823044e-01,  8.40896648e+01],
    #        [ 1.90286866e-04, -3.32648501e-04,  1.00000000e+00]])
    
    # Far matrix:
    M = np.array([[ 6.13066698e-01, -5.12163248e-02,  1.31529098e+02],
           [ 1.14860326e-02,  1.04269100e+00,  5.27519449e+00],
           [ 5.08809793e-05, -1.02988410e-04,  1.00000000e+00]])        
    
    
    while True:
            
        # Capture synchronized frames from each camera
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
        #ret3, frame3 = cap3.read()

        if not ret1 or not ret2: # or not ret3:
            break

        # Get the timestamp of the first camera as reference
        reference_timestamp = cap1.get(cv2.CAP_PROP_POS_MSEC)

        # Synchronize the other cameras' frames to the reference timestamp
        while cap2.get(cv2.CAP_PROP_POS_MSEC) < reference_timestamp:
            cap2.grab()

      #  while cap3.get(cv2.CAP_PROP_POS_MSEC) < reference_timestamp:
      #      cap3.grab()

        # Resize frames to the common resolution
        frame1 = cv2.resize(frame1, (common_width, common_height))
        frame2 = cv2.resize(frame2, (common_width, common_height))
      # frame3 = cv2.resize(frame3, (common_width, common_height))

        ## Using align_frames function
        # Align frames based on frame2 (reference frame)
        #aligned_frame1 = align_frames_orb(frame1, frame2)
        #aligned_frame3 = align_frames_orb(frame3, frame2)

        
        ## Using Calibrated M
        aligned_frame1 = cv2.warpPerspective(frame1, M, (frame2.shape[1], frame2.shape[0]))

        # Process and overlay frames to create composite
        # Two cameras
        composite_frame = cv2.addWeighted(aligned_frame1, 0.5, frame2, 0.5, 0)
        # Three cameras
        #composite_frame = cv2.addWeighted(aligned_frame1, 0.33, frame2, 0.33, 0)
        #composite_frame = cv2.addWeighted(composite_frame, 1, aligned_frame3, 0.33, 0)

        _, buffer = cv2.imencode('.jpg', composite_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)