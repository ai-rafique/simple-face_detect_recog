import cv2
from handle_image_data import get_image_data
from create_image_data import make_image_data_of
from camera import facial_recognition
import time

print('Welcome\n')
names = []
cascade_file = 'haarcascade_frontalface_default.xml'
lbph_file = 'lbph_2_face_classifier.yml'

# Enter name of the people you want to recognise
name1 = input('Enter name of the first person :')
name2 = input('Enter name of the second person :')
names.append(name1)
names.append(name2)

# Lets make image data for the two
for i in range(len(names)):
    make_image_data_of(names[i], i + 1, 10)
    time.sleep(5)

# Train the LBPH classifier
ids, faces = get_image_data()
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write(lbph_file)

# Lets detect and recognize the faces
facial_recognition(cascade_file, lbph_file, names)
