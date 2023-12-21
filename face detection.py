from mtcnn import MTCNN
from google.colab.patches import cv2_imshow
import cv2

# Create MTCNN detector
detector = MTCNN()

# Read the image
img = cv2.imread('/content/drive/MyDrive/vk.jpg')

# Convert the image to RGB (MTCNN expects RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces
output = detector.detect_faces(img_rgb)

# Loop over the detected faces and keypoints
for i in output:
    x, y, width, height = i['box']

    left_eyeX, left_eyeY = i['keypoints']['left_eye']
    right_eyeX, right_eyeY = i['keypoints']['right_eye']
    noseX, noseY = i['keypoints']['nose']
    mouth_leftX, mouth_leftY = i['keypoints']['mouth_left']
    mouth_rightX, mouth_rightY = i['keypoints']['mouth_right']

    # Draw circles at keypoints
    cv2.circle(img, center=(left_eyeX, left_eyeY), color=(255, 0, 0), thickness=3, radius=2)
    cv2.circle(img, center=(right_eyeX, right_eyeY), color=(255, 0, 0), thickness=3, radius=2)
    cv2.circle(img, center=(noseX, noseY), color=(255, 0, 0), thickness=3, radius=2)
    cv2.circle(img, center=(mouth_leftX, mouth_leftY), color=(255, 0, 0), thickness=3, radius=2)
    cv2.circle(img, center=(mouth_rightX, mouth_rightY), color=(255, 0, 0), thickness=3, radius=2)

    # Draw rectangle around the face
    cv2.rectangle(img, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=3)

# Display the image using cv2_imshow
cv2_imshow(img)



