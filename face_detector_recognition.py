import cv2
from face_recognition_models import face_recognition_model_location
import dlib
import numpy as np
import face_recognition
import glob

detector = dlib.get_frontal_face_detector()

def MyRec(rgb,x,y,w,h,v=25,color=(200,0,0),thikness =2):
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
def load_and_save_img(filename,counter,path):
    
    # Load the input image
    frame =cv2.imread(filename)
    image =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = detector(image)
    fit =20
    # detect the face
    #if len(faces)>1:

    for counter1,face in enumerate(faces):
        print(counter)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
        MyRec(frame, x1, y1, x2 - x1, y2 - y1, 1, (0,250,0), 3)
        #save(image,new_path+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
        new_path = f'{path}_{counter}_{counter1}'
        save(frame,new_path,(x1,y1,x2,y2))
        #frame = cv2.resize(frame,(800,800))
        cv2.imshow('img',frame)
        cv2.waitKey(0)
        print("done saving")

        


def encode_image(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    encoded_image = face_recognition.face_encodings(img)[0]

    # print(encoded_image)
    return encoded_image

# Store the encoded vec in a dictionary
count = 0
counter = 9
encoded_vec = {}
encoded_vec_list = []
image_vec = {}

for image in glob.glob('images/Train' + '/' + '*.*'):
    print(image)
    #count = count + 1
    counter= counter + 1
    load_and_save_img(image,counter,'images/extracted_images/')
    frame =cv2.imread(image)
    img =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = detector(img)
    print(len(faces))
    if len(faces)>1:
        for i in range(len(faces)):
            print('saved_twice')
            count = count+1
            image_vec[count] = image
    else:
      count = count + 1
      image_vec[count] = image
for img in glob.glob('images/extracted_images' + '/' + '*.*'):
    print(img)
    try:
      img_encoding = encode_image(img)
    except:
        continue
    encoded_vec_list.append(img_encoding)



print(encoded_vec_list)


# Testing our encodings

#test_path = load_and_save_img('images\Train/apj.jpg',20)
test_path = 'images\images_for_extraction/apj(3).jpg0.jpg0.jpg'
count_test = 0
counter_test = 9
encoded_vec_list_test = []
counter_test= counter_test + 1
frame_test =cv2.imread(test_path)
img_test =cv2.cvtColor(frame_test,cv2.COLOR_BGR2RGB)
faces_test = detector(img_test)
print(len(faces_test))
if len(faces_test)>1:
    results = []
    load_and_save_img(test_path,counter_test,'images/test_extracted_images/')
    for img_test in glob.glob('images/test_extracted_images' + '/' + '*.*'):
        try:
            img_encoded_test = encode_image(img_test)
        except:
            continue
        results = face_recognition.compare_faces(encoded_vec_list,img_encoded_test,tolerance=0.6)
        print(results)
        image_test = cv2.imread(test_path)
        cv2.imshow('Image',image_test)
        cv2.waitKey(0)
        for i in range(len(encoded_vec_list)):
            if results[i] == True:
                image = cv2.imread(image_vec[i+1])
                cv2.imshow('Image',image)
                cv2.waitKey(0)

else:
    img_encoding_test = encode_image(test_path)




    # Compare
    results = face_recognition.compare_faces(encoded_vec_list,img_encoding_test,tolerance=0.6)
    print(results)

    image_test = cv2.imread(test_path)
    cv2.imshow('Image',image_test)
    cv2.waitKey(0)



    for i in range(len(encoded_vec_list)):
        if results[i] == True:
            image = cv2.imread(image_vec[i+1])
            cv2.imshow('Image',image)
            cv2.waitKey(0)
