import os
import argparse

from face_alignment import image_align
from landmarks_detector import LandmarksDetector
import cv2

class Align_image:
    def __init__(self):
        self.landmarks_model_path='models/shape_predictor_68_face_landmarks.dat'
    def align(self,img):
        # file_name, file_ext = os.path.splitext(os.path.basename(args.input))
        landmarks_detector = LandmarksDetector(self.landmarks_model_path)

        try:
            all_face_landmarks = landmarks_detector.get_landmarks(img)
            for i, face_landmarks in enumerate(all_face_landmarks):
                image = image_align(img, face_landmarks)
                return image
        except Exception as e:
            print("Error:", e)
