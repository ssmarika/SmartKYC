import cv2
import numpy as np

# Function to resize image to fit within the screen size
def resize_image(image, width=None, height=None):
    # Get the original dimensions of the image
    h, w = image.shape[:2]

    # If both width and height are None, do nothing
    if width is None and height is None:
        return image

    # Calculate the ratio of the original dimensions to the desired dimensions
    if width is not None:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    else:
        ratio = height / float(h)
        dim = (int(w * ratio), height)

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# Load the template and uploaded images
template_image = cv2.imread(r'C:\Users\uSer\Desktop\SmartKYC\Template Matching\img4.jpg', cv2.IMREAD_GRAYSCALE)
uploaded_image = cv2.imread(r'C:\Users\uSer\Desktop\SmartKYC\Template Matching\uploaded_document.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect key points and compute descriptors
keypoints_template, descriptors_template = sift.detectAndCompute(template_image, None)
keypoints_uploaded, descriptors_uploaded = sift.detectAndCompute(uploaded_image, None)

# Use FLANN based matcher to find matches between the descriptors
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Perform matching between template and uploaded document
matches = flann.knnMatch(descriptors_template, descriptors_uploaded, k=2)

# Filter good matches using Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Minimum number of good matches required
MIN_MATCH_COUNT = 100

if len(good_matches) > MIN_MATCH_COUNT:
    print(f"Match found! Number of good matches: {len(good_matches)}")

    # Draw matches between the template and uploaded document
    matched_image = cv2.drawMatches(template_image, keypoints_template, uploaded_image, keypoints_uploaded, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Resize the matched image to fit the screen size (for example, width=1000 pixels)
    matched_image_resized = resize_image(matched_image, width=1000)

    # Display the resized image
    cv2.imshow("Matched keypoints (resized)", matched_image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print(f"Not a valid driving license. Only {len(good_matches)} matches found.")
