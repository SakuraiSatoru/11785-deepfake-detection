#!/bin/sh
# Sets up the data
#  - Download only original and Deepfakes at c40
#  - Perform face crop and face alignment using dlib (NOT USING MASK)
# NOTE: download, extract frame and frame preprocessing don't run if output directory exists
#   be careful (each video directory for extract frame and frame preprocessing)

# Before running do the following:
# Set up preprocessing environment
# conda create --name preprocessing
# source activate preprocessing
# conda install -c conda-forge opencv
# pip install tqdm
# pip install dlib
# pip install notify_run

# Download face landmark detection model
# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Create data directory
mkdir ../data
mkdir ../data/faceforensics
mkdir ../data/faceforensics/c40

DATA_DIR="../data/faceforensics/c40"
NUM_VIDS=1000
COMPRESSION="c40"
FILE_NAME="setup_data.sh"
FAKE_TYPE="Deepfakes"

# Download data
echo "${FILE_NAME}: Downloading data"
python faceforensics_download_v4.py -d original -c $COMPRESSION -n $NUM_VIDS $DATA_DIR
python faceforensics_download_v4.py -d Deepfakes -c $COMPRESSION -n $NUM_VIDS $DATA_DIR

# Extract frames
# echo "${FILE_NAME}: Extracting frames"
# python extract_frames_from_videos.py --data_path $DATA_DIR -d original -c $COMPRESSION 
# python extract_frames_from_videos.py --data_path $DATA_DIR -d $FAKE_TYPE -c $COMPRESSION 

# Perform face crop and alignment
# echo "${FILE_NAME}: Perform face crop and alignment"
# python face_crop_and_align_frames.py --data_path $DATA_DIR --landmark_model_path shape_predictor_68_face_landmarks.dat -d original
# python face_crop_and_align_frames.py --data_path $DATA_DIR --landmark_model_path shape_predictor_68_face_landmarks.dat -d $FAKE_TYPE

# Extract frames and preprocess
echo "${FILE_NAME}: Extracting frames and preprocessing"
python extract_and_preprocess_videos.py --data_path $DATA_DIR -c $COMPRESSION -d original 
python extract_and_preprocess_videos.py --data_path $DATA_DIR -c $COMPRESSION -d $FAKE_TYPE