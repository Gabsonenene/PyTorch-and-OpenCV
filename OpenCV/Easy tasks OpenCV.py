# Loading libraries
import cv2
import numpy as np

# Task 1
# Loading and displaying an image
image = cv2.imread("turtle.jpg")
cv2.imshow("Turtle", image)
cv2.waitKey(0) 

# Task 2
# Changing to a shade of grey
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)

# Task 3 
# Cropping the image
print(image.shape)
cropped_image = image[200:600, 400:1100]
cv2.imshow("cropped", cropped_image)
cv2.waitKey(0)

# Rotate 90 degrees
rotated = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("rotated", rotated)
cv2.waitKey(0)

# Task 4 
# Save the pictures
filename1 = 'turtlegray.jpg'
filename2 = 'turtle90.jpg'

cv2.imwrite(filename1, gray) 
cv2.imwrite(filename2, rotated)
print('Successfully saved') 

# Task 5
# Edge detection
blur = cv2.GaussianBlur(gray, (3,3), 0) # blurring the image

# Three different methods
otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
triangle_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_TRIANGLE)
manual_thresh = np.median(blur)

def get_range(threshold, sigma = 0.33):
    return (1-sigma) * threshold, (1+sigma) * threshold

otsu_thresh = get_range(otsu_thresh)
triangle_thresh = get_range(triangle_thresh)
manual_thresh = get_range(manual_thresh)

edge_otsu = cv2.Canny(blur, *otsu_thresh)
edge_triangle = cv2.Canny(blur, *triangle_thresh)
edge_manual = cv2.Canny(blur, *manual_thresh)

cv2.imshow("Triangle", edge_triangle)
cv2.waitKey(0)

cv2.imshow("Otsu", edge_otsu)
cv2.waitKey(0)

cv2.imshow("Manual", edge_manual)
cv2.waitKey(0)

# Task 6
# Face detection
img = cv2.imread("barbie.jpg") # changed the photo
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
faces = face_detector.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_detector.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img', img)
cv2.waitKey(0)