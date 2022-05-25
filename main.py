import cv2
import torch
import numpy as np
import mediapipe as mp
from data_to_graph import build_graph, norm_points
from model import GNNStack
from torch_geometric.loader import DataLoader

model = torch.load('./models/0.9517.pth')
model.eval()

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

'''
Stream
'''
frame_count = 0  # current frame counter

'''
points
'''
points_arr = np.zeros((478, 2))  # initialize an array to contain landmarks points

# camera stream:
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmark in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:  # no frame input
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            frame_count = frame_count + 1

            for i in range(478):
                points_arr[i][0] = results.multi_face_landmarks[0].landmark[i].x
                points_arr[i][1] = 1 - results.multi_face_landmarks[0].landmark[i].y
            arr = norm_points(points_arr)
            graph = build_graph(arr, 0)
            loader = DataLoader([graph, graph], batch_size=2)
            batch = next(iter(loader))
            flag = model(batch).argmax(dim=1)
            if torch.sum(flag) == 2:
                image = cv2.putText(image, 'Happy', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5, cv2.LINE_AA)
            else:
                image = cv2.putText(image, 'Sad', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imshow('Smile', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
    cap.release()