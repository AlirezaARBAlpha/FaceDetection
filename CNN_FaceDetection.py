import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np

gray = cv2.imread('Enter url image', 0)
im = np.float32(gray) / 255.0
# Calculate gradient 
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

dnnFaceDetector = dlib.cnn_face_detection_model_v1("data/mmod_human_face_detector.dat")
rects = dnnFaceDetector(gray, 0)
for (i, rect) in enumerate(rects):
    x1 = rect.rect.left()
    y1 = rect.rect.top()
    x2 = rect.rect.right()
    y2 = rect.rect.bottom()
    # Rectangle around the face
    cv2.rectangle(gray, (x1, y1), (x2, y2), (0, 128, 0), 3)
plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()
