import cv2
import os


def make_image_data_of(name, label, count):
    """
    Creates a training folder if not exists
    creates a named.classnum.index.png image
    Stores them in training folder

    """
    camera = cv2.VideoCapture(0)
    i = 0
    if not os.path.exists('training'):
        os.mkdir('training')

    while i < count:
        return_value, image = camera.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'training/{name}.{label}.' + str(i) + '.png', gray)
        i += 1
    del camera

