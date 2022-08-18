import os
import argparse

from face_alignment import image_align
from landmarks_detector import LandmarksDetector
import cv2


def align(img):
    landmarks_detector = LandmarksDetector('models/shape_predictor_68_face_landmarks.dat')
    # file_name, file_ext = os.path.splitext(os.path.basename(args.input))

    try:
        all_face_landmarks = landmarks_detector.get_landmarks(img)
        for i, face_landmarks in enumerate(all_face_landmarks):
            image,crop = image_align(img, face_landmarks)
            return image

    except Exception as e:
        print("Error:", e)



