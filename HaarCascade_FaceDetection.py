import cv2
import matplotlib.pyplot as plt

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

gray = cv2.imread('/content/serious-young-african-man-standing-isolated.jpg', 1)

# Detect faces
faces = faceCascade.detectMultiScale(
gray,
scaleFactor=1.1,
minNeighbors=5,
flags=cv2.CASCADE_SCALE_IMAGE
)
# For each face
for (x, y, w, h) in faces: 
    # Draw rectangle around the face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = gray[y:y+h, x:x+w]
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 128, 0), 7)
    
smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.2,
        minNeighbors=255,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

for (sx, sy, sw, sh) in smile:
      cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 5)
      cv2.putText(gray,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)

eyes = eyeCascade.detectMultiScale(roi_gray)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.putText(gray,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()
