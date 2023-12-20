
import cv2
from google.colab import files
from IPython.display import Image
Choose files gwenn.jpg
gwenn.jpg(image/jpeg) - 207029 bytes, last modified: 14/11/2023 - 100% done
Saving gwenn.jpg to gwenn.jpg
# Upload an image file
uploaded = files.upload()
# Get the file name
file_name = list(uploaded.keys())[0]
# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Read the uploaded image
img = cv2.imread(file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Display the result
cv2.imwrite('result.jpg', img)
Image(filename='result.jpg')
