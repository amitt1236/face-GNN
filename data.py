from math import pi
import numpy as np
import cv2
import os
import mediapipe as mp
import datetime
import gaze 

mp_face_mesh = mp.solutions.face_mesh
points_arr = np.zeros((478, 2))

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for folder in range(2): # for 2 classes
        for filename in os.listdir("./raw_data/" + str(folder)):
            if filename[0] == '.':
                continue

            image = cv2.imread("./raw_data/" + str(folder) + '/' + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            shape = np.array([[image.shape[1], 0],[0, image.shape[0]]])
            rotation_matrix = lambda x : np.array([[np.cos(pi/100 * x), -np.sin(pi/100 * x)],[np.sin(pi / 100 * x), np.cos(pi /100 * x)]])
            if results.multi_face_landmarks:
                for i in range(478):
                    points_arr[i][0] = results.multi_face_landmarks[0].landmark[i].x
                    points_arr[i][1] = 1 - results.multi_face_landmarks[0].landmark[i].y    
                
                # adding gaze point, replace point 10 (not in use)
                points_arr[10] = gaze.gaze(image, points_arr @ shape)
               
                # applying rotation augmentation, each step is 1.8 degrees
                for r in range(-3, 4):  
                    shift_points_arr = [rotation_matrix(r) @ R.T for R in points_arr]
                    # applying 4 gussian noise augmentation
                    for _ in range(4): 
                        mu, sigma = 0, 0.0001 # mean and standard deviation
                        gus_noise = np.random.normal(mu, sigma, [478,2])
                        shift_points_arr_noise = shift_points_arr + gus_noise
                        np.save("points_data/" + str(folder) + '/' + str(datetime.datetime.now()) + ".npy", shift_points_arr_noise)