import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from numpy import expand_dims
from scipy.spatial.distance import euclidean

def preprocess_image(image_path):
    """Load and preprocess image for FaceNet model."""
    image = cv2.imread(image_path)
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

def load_facenet_model(model_path):
    """Load the pre-trained FaceNet model."""
    model = load_model(model_path)
    return model

def get_embedding(model, face):
    """Get the embedding of a face image."""
    embedding = model.predict(face)
    return embedding[0]

def is_same_person(embedding1, embedding2, threshold=1.0):
    """Determine if two face embeddings are from the same person."""
    distance = euclidean(embedding1, embedding2)
    return distance < threshold

# Load the pre-trained FaceNet model (make sure to download facenet_keras.h5)
model_path = r'C:\Users\uSer\Desktop\Major\FaceNet\facenet_keras.h5'  # Adjust this path accordingly
model = load_facenet_model(model_path)

# Paths to the images to be compared
image_path1 = r'C:\Users\uSer\Desktop\Major\FaceNet\img1.jpg'
image_path2 = r'C:\Users\uSer\Desktop\Major\FaceNet\img2.img2.jpg'

# Preprocess the images
face1 = preprocess_image(image_path1)
face2 = preprocess_image(image_path2)

# Get the embeddings
embedding1 = get_embedding(model, face1)
embedding2 = get_embedding(model, face2)

# Determine if they are the same person
result = is_same_person(embedding1, embedding2)
print(f"Are the images of the same person? {'Yes' if result else 'No'}")

