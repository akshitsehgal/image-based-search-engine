import cv2
from face_recognition_models import face_recognition_model_location
import dlib
import numpy as np
import face_recognition
import glob

detector = dlib.get_frontal_face_detector()

def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =2):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

def save(img,name, bbox, width=180,height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))#we need this line to reshape the images
    cv2.imwrite(name+".jpg", imgCrop)


# Define a function that loads and encodes an image
def load_and_encode_img(filename):
    '''
    Loads in an image from it's directory using 
    face_recognition.load_image_file method and returns it's 
    face encodings.
    '''
    
    # Load the input image
    frame =cv2.imread(filename)
    image =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = detector(image)
    fit =20
    # detect the face
    for counter,face in enumerate(faces):
        print(counter)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
        MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
        #save(image,new_path+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
        new_path = f'images/extracted_images/detector_recog_model{counter}'
        save(frame,new_path,(x1,y1,x2,y2))
        #frame = cv2.resize(frame,(800,800))
        cv2.imshow('img',frame)
        cv2.waitKey(0)
        print("done saving")
    for img in glob.glob('images/extracted_images' + '/' + '*.jpg'):
        img = cv2.imread(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoded_image = face_recognition.face_encodings(img)[0]
        print(encoded_image)

    return encoded_image

# Store the encoded vec in a dictionary
count = 0
encoded_vec = {}
encoded_vec_list = []
image_vec = {}

for image in glob.glob('images/Train' + '/' + '*.jpg'):
    print(image)
    count = count + 1
    img_encoding = load_and_encode_img(image)
    #encoded_vec[count] = img_encoding
    image_vec[count] = image
    encoded_vec_list.append(img_encoding)


print(encoded_vec_list)


# Testing our encodings

encoded_img_test = load_and_encode_img('images\Test\depositphotos_29133839-stock-photo-group-of-girls-chilling-on.jpg')


# Compare
results = face_recognition.compare_faces(encoded_vec_list,encoded_img_test)
print(results)

for i in range(len(encoded_vec_list)):
    if results[i] == True:
        image = cv2.imread(image_vec[i+1])
        cv2.imshow('Image',image)
        cv2.waitKey(0)
