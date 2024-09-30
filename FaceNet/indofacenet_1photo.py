import os
from PIL import Image
from numpy import asarray
from numpy import expand_dims
import numpy as np
import cv2
from keras_facenet import FaceNet

# Load Haar Cascade Classifier for face detection
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Load FaceNet model using keras_facenet
MyFaceNet = FaceNet()

def generate_face_signature(image_path):
    # Read the image using OpenCV
    gbr1 = cv2.imread(image_path)
    
    # Detect faces in the image using Haar Cascade
    faces = HaarCascade.detectMultiScale(gbr1, 1.1, 4)
    
    if len(faces) > 0:
        x1, y1, width, height = faces[0]         
    else:
        raise ValueError("No face detected in the image.")
        
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # Convert the image from OpenCV format to PIL format
    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Image.fromarray(gbr)
    gbr_array = asarray(gbr)
    
    # Extract and resize the face region
    face = gbr_array[y1:y2, x1:x2]
    face = Image.fromarray(face)
    face = face.resize((160, 160))
    face = asarray(face)
    
    # Expand dimensions to match FaceNet input
    face = expand_dims(face, axis=0)
    
    # Get face embedding using FaceNet
    signature = MyFaceNet.embeddings(face)
    
    return signature

# Path to the image to be processed
image_path = r'C:\Users\uSer\Desktop\Major\FaceNet\img1.jpg'

# Generate and print the face signature
signature = generate_face_signature(image_path)
print(signature)
