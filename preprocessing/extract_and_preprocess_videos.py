"""
Extracts images and preprocesses them from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h

Author: Nathan Kuo (modified from extract_frames_from_videos.py)
"""


import sys
import os
from os.path import join
import argparse
import subprocess
import random
import cv2
import dlib
from tqdm import tqdm
from notify_run import Notify

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']
CROP_SIZE = 224
CROP_PADDING = 0.25

def preprocess_and_save_image(img, face_detector, landmark_model, output_path, frame_num):
    detections = face_detector(img, 1) # upscale by 1
    if len(detections) == 0:
        print("{}: face not detected on frame {}, skipping...".format(output_path.split("/")[-1], frame_num))
        # Save full image if face not detected
        outpath = join(output_path, "face_not_detected")
        os.makedirs(outpath, exist_ok=True)
        dlib.save_image(img, join(outpath, "{:04d}.png".format(frame_num)))
        return

    # Keep only one face for now
    index = 0
    # for index in range(len(detections)):
    detection = detections[index]
    landmarks = landmark_model(img, detection)
    processed_image = dlib.get_face_chip(img, landmarks, size=CROP_SIZE, padding=CROP_PADDING)
    dlib.save_image(processed_image, join(output_path, "{:04d}.png".format(frame_num)))

def extract_frames(data_path, output_path, landmark_model_path, skip_if_dir_exists):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    if skip_if_dir_exists:
        if os.path.exists(output_path): # if output directory exists
            print(output_path + " exists, skipping frames")
            return
    if output_path.split("/"[-1]) in FAKES_SKIP:
        print("FAKES_SKIP.contains({})".format(output_path.split("/"[-1])))
        return

    # load models
    face_detector = dlib.get_frontal_face_detector()
    landmark_model = dlib.shape_predictor(landmark_model_path)
    
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    frame_num = 0

    while reader.isOpened():
        success, image = reader.read()
        if not success: # If reach end of video
            break
        # video captures in BGR, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # check if image exists already
        if not os.path.exists(join(output_path, "{:04d}.png".format(frame_num))):
            preprocess_and_save_image(image, face_detector, landmark_model, output_path, frame_num)
        else:
            if skip_if_dir_exists:
                return
            if frame_num % 100 == 0:
                print("{}: frame {} exists, skipping...".format(output_path.split("/")[-1], frame_num))

        frame_num += 1
    reader.release()


def extract_method_videos(data_path, dataset, compression, landmark_model_path, skip_if_dir_exists):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'processed_images')
    videos = os.listdir(videos_path)
    random.shuffle(videos)
    for video in tqdm(videos):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder), landmark_model_path, skip_if_dir_exists)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    p.add_argument('--landmark_model_path', type=str, default='shape_predictor_68_face_landmarks.dat')
    p.add_argument('--skip_if_dir_exists', action='store_true')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))

    notify = Notify(endpoint="https://notify.run/Dbnkja3hR3rG7MuV")
    notify.send(" ".join(sys.argv) + " done")
