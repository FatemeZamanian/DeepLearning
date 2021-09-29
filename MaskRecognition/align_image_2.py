import os
import argparse

from imutils.face_utils import FaceAligner
import imutils
import dlib
import cv2


class Align_image:
    def __init__(self):
        self.landmarks_model_path='models/shape_predictor_68_face_landmarks.dat'
    def align(self,img):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.landmarks_model_path)
        fa = FaceAligner(predictor, desiredFaceWidth=256)

    # image = cv2.imread(args.input)
        image = imutils.resize(img, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 2)

        for i, rect in enumerate(rects):
            faceAligned = fa.align(image, gray, rect)
        
            return faceAligned
