import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from numpy import expand_dims
from scipy.spatial.distance import euclidean

def preprocess_image(image_path):
    """Load and preprocess image for FaceNet model."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unable to read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        faces = detector.detect_faces(image)
        if len(faces) == 0:
            raise ValueError("No face detected in the image.")
        x, y, width, height = faces[0]['box']
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        face = expand_dims(face, axis=0)
        return face
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise

def load_facenet_model(model_path):
    """Load the pre-trained FaceNet model."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error in load_facenet_model: {e}")
        raise

def get_embedding(model, face):
    """Get the embedding of a face image."""
    try:
        embedding = model.predict(face)
        return embedding[0]
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        raise

def is_same_person(embedding1, embedding2, threshold=1.0):
    """Determine if two face embeddings are from the same person."""
    distance = euclidean(embedding1, embedding2)
    return distance < threshold

# Load the pre-trained FaceNet model (make sure to download facenet_keras.h5)
model_path = r'C:\Users\uSer\Desktop\SmartKYC\FaceNet\facenet_keras.h5'  # Adjust this path accordingly
print(f"Loading model from {model_path}")
model = load_facenet_model(model_path)

# Paths to the images to be compared
image_path1 = r'C:\Users\uSer\Desktop\SmartKYC\FaceNet\img1.jpg'
image_path2 = r'C:\Users\uSer\Desktop\SmartKYC\FaceNet\img2.jpg'  # Corrected file extension

# Preprocess the images
print(f"Preprocessing image 1: {image_path1}")
face1 = preprocess_image(image_path1)
print(f"Preprocessing image 2: {image_path2}")
face2 = preprocess_image(image_path2)

# Get the embeddings
print("Getting embeddings")
embedding1 = get_embedding(model, face1)
embedding2 = get_embedding(model, face2)

# Determine if they are the same person
print("Comparing embeddings")
result = is_same_person(embedding1, embedding2)
print(f"Are the images of the same person? {'Yes' if result else 'No'}")
