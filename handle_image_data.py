import numpy as np
import os
from PIL import Image


def get_image_data():
    paths = [os.path.join('training', f) for f in os.listdir('training')]
    faces = []
    ids = []
    for path in paths:
        image = Image.open(path).convert('L')
        image_np = np.array(image, 'uint8')
        id = int(path.split('.')[1])

        ids.append(id)
        faces.append(image_np)

    return np.array(ids), faces
