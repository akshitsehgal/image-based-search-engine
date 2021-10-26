import cv2
import mediapipe as mp

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh()


# Image
IMAGE_PATH = 'images/Train/apj.jpg'
image = cv2.imread(IMAGE_PATH)
# Get the height and width of the image
height, width , _ = image.shape
print(f'Height: {height}\n Width: {width}')
rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Facial Landmarks
result = face_mesh.process(image)

for facial_landmarks in result.multi_face_landmarks:
  for i in range(0,468):
      
      pt = facial_landmarks.landmark[i]
      #print(pt)
      x = int(pt.x * width)
      y = int(pt.y * height)
      print(x,y)
  
      cv2.circle(img = image, 
               center = (x,y), 
               radius = 1, 
               color = (100,0,100),
               thickness= -1)

cv2.imshow("Image",image)
cv2.waitKey(0)
