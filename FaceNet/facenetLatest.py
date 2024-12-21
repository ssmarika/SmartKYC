import sys
import cv2
import numpy as np
from keras_facenet import FaceNet
from PIL import Image as Img
from numpy import asarray, expand_dims

# Force UTF-8 encoding for script output
sys.stdout.reconfigure(encoding='utf-8')

# Initialize FaceNet model
MyFaceNet = FaceNet()

# Load Haar Cascade for face detection
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Function to extract face embedding from an image
def get_embedding(image_path):
    try:
        # Log the file being processed
        print(f"Processing file: {image_path}")

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found or unreadable at path: {image_path}")

        # Detect face
        faces = HaarCascade.detectMultiScale(img, 1.1, 4)
        if len(faces) == 0:
            raise ValueError("No face detected in the image: " + image_path)

        # Get the first face
        x1, y1, width, height = faces[0]
        x2, y2 = x1 + width, y1 + height

        # Extract face region
        face = img[y1:y2, x1:x2]

        # Resize and preprocess face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Img.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)
        face = expand_dims(face, axis=0)

        # Generate embedding
        embedding = MyFaceNet.embeddings(face)
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to extract embedding: {e}")

# Function to compare two embeddings
def verify_faces(embedding1, embedding2, threshold=0.6):
    # Compute Euclidean distance between embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    print(f"Euclidean Distance: {distance}")

    # Return whether the distance is below the threshold
    return distance < threshold

# Input images
image1 = r'C:\Users\uSer\Desktop\SmartKYC\FaceNet\img1.jpg'  # Replace with path to first image
image2 = r'C:\Users\uSer\Desktop\SmartKYC\FaceNet\img2.jpg'  # Replace with path to second image

try:
    # Get embeddings
    embedding1 = get_embedding(image1)
    embedding2 = get_embedding(image2)

    # Verify if they are the same person
    result = verify_faces(embedding1, embedding2)
    if result:
        print("The two photos belong to the same person.")
    else:
        print("The two photos do NOT belong to the same person.")
except Exception as e:
    print(f"Error: {e}")

