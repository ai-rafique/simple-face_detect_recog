# simple-face_detect_recog
This is a simple facial detection and recognition script using the Local Binary Pattern Histogram or LBPH Algorithm in tandem with the Haar frontal facial cascade classifier to detect a face.

The main function starts by asking the user to enter 2 names (Keeping things simple, we will work with 2 people only)
The name is sent to the image data creator script which will make a training folder with the images being stored in 

name.classnum.index.png format.

Then the LBPH classifier will train on those images producing a .yml file.
In the recognition section, the Haar cascade will be used for detection and the features learnt in the lbph yml will help detect the images.

Please note that the image data maker is one function at the moment so to detect 2 faces, there is a delay of 5 seconds so when the camera takes the first set
of images, within 5 seconds, the other person must be infront of the camera.

This project was done mainly to check out how lbph works. There are plenty of better ways than this available afterall.
