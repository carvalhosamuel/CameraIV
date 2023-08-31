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

from flirpy.camera.lepton import Lepton

with Lepton() as camera:
    while True:
        img = camera.grab().astype(np.float32)

        # Rescale to 8 bit
        img = 255*(img - img.min())/(img.max()-img.min())

        # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
        # You can also try PLASMA or MAGMA
        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)

        cv2.imshow('Boson', img_col)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

cv2.destroyAllWindows()







app = Flask(__name__)

# Initialize capture from each camera
cap1 = cv2.VideoCapture(0)  # Camera 1
cap2 = cv2.VideoCapture(2)  # Camera 2
cap3 = cv2.VideoCapture(4)  # Camera 3

def get_cameras(up_to):
    
    for i in range(up_to):
        
        try:
            cap = cv2.VideoCapture(i)
           
            result, image = cap.read()
      
            if result:
                # Using cv2.imshow() method
                # Displaying the image
                cv2.imshow(str(i), image)
                #time.sleep(1)
                # waits for user to press any key
                # (this is necessary to avoid Python kernel form crashing)
                cv2.waitKey(0)
              
                # closing all open windows
                cv2.destroyAllWindows()
        except:
            print("Camera" + i +" not present")
            
            


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
        
        return aligned_frame
    
    except:
        return frame1







def generate_frames():
    while True:
        # Capture synchronized frames from each camera
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if not ret1 or not ret2 or not ret3:
            break

        # Get the timestamp of the first camera as reference
        reference_timestamp = cap1.get(cv2.CAP_PROP_POS_MSEC)

        # Synchronize the other cameras' frames to the reference timestamp
        while cap2.get(cv2.CAP_PROP_POS_MSEC) < reference_timestamp:
            cap2.grab()

        while cap3.get(cv2.CAP_PROP_POS_MSEC) < reference_timestamp:
            cap3.grab()

        # Resize frames to the common resolution
        frame1 = cv2.resize(frame1, (common_width, common_height))
        frame2 = cv2.resize(frame2, (common_width, common_height))
        frame3 = cv2.resize(frame3, (common_width, common_height))

        # Align frames based on frame2 (reference frame)
        aligned_frame1 = align_frames_orb(frame1, frame2)
        aligned_frame3 = align_frames_orb(frame3, frame2)

        # Process and overlay frames to create composite
        # Two cameras
        #composite_frame = cv2.addWeighted(aligned_frame1, 0.5, frame2, 0.5, 0)
        # Three cameras
        composite_frame = cv2.addWeighted(aligned_frame1, 0.33, frame2, 0.33, 0)
        composite_frame = cv2.addWeighted(composite_frame, 1, aligned_frame3, 0.33, 0)

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