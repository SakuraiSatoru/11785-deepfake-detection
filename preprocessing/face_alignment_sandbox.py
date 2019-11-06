#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool for image alignment.
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys

import dlib

if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   ./face_alignment.py shape_predictor_5_face_landmarks.dat ../examples/faces/bald_guys.jpg\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")
    exit()

predictor_path = sys.argv[1]
face_file_path = sys.argv[2]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Load the image using Dlib
img = dlib.load_rgb_image(face_file_path)

dets = detector(img, 1) # image and upsample the size of image by 1

num_faces = len(dets)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()

# Display image with landmarks detected
window = dlib.image_window()
window.set_image(img)
faces = dlib.full_object_detections()
for detection in dets:
    shape = sp(img, detection)
    faces.append(shape)

    window.add_overlay(shape)
print("Display full image with landmark overlay")
# dlib.hit_enter_to_continue()
window.clear_overlay()

# Align face images based on landmarks
images_with_padding = dlib.get_face_chips(img, faces, size=224, padding=0.25) #0.30, higher = looser crop
# print("Show aligned images")
images = dlib.get_face_chips(img, faces, size=224)
# for image in images:
#     window.set_image(image)
#     dlib.hit_enter_to_continue()

window_padded = dlib.image_window()

# print("Detect landmarks again")
print("Comparing padded vs non-padded")
for i in range(len(images)):
    image = images[i]
    image_padded = images_with_padding[i]

    window.set_image(image)
    window_padded.set_image(image_padded)

    # Recompute landmarks and display with image
    detections = detector(image, 2)
    window.clear_overlay()

    window.set_image(image)
    for detection in detections:
        shape = sp(image, detection)
        window.add_overlay(shape)

    detections = detector(image_padded, 2)
    window_padded.clear_overlay()

    window_padded.set_image(image_padded)
    for detection in detections:
        shape = sp(image_padded, detection)
        window_padded.add_overlay(shape)
    
    dlib.hit_enter_to_continue()