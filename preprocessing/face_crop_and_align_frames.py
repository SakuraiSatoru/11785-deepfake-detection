import os
from os.path import join
import sys
import dlib
import argparse
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

def crop_and_align_images(image_path, output_path, landmark_model_path):
    """Method to crop and align each image in img_path and output into processed_img_path"""
    if os.path.exists(output_path): # if output directory exists
        print(output_path + " exists, skipping")
        return
        
    os.makedirs(output_path, exist_ok=True)

    # Create face detector and landmark detector
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(landmark_model_path)
    for image_name in tqdm(os.listdir(image_path)): # for each image
        # Load image and process
        img = dlib.load_rgb_image(join(image_path, image_name))
        image_name = image_name.split(".")
        detections = detector(img, 1)

        for index in range(len(detections)):
            detection = detections[index]
            landmarks = sp(img, detection)
            processed_img = dlib.get_face_chip(img, landmarks, size=CROP_SIZE, padding=CROP_PADDING)
            dlib.save_image(processed_img, join(output_path, "{}_{}.{}".format(image_name[0], str(index), image_name[1])))

def crop_and_align_all(data_path, dataset, compression, landmark_model_path):
    """Crop and align all images of specified method and compression in the
    FaceForensics++ file structure"""
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    processed_images_path = join(data_path, DATASET_PATHS[dataset], compression, 'processed_images')
    for video_name in tqdm(os.listdir(images_path)): # for each video
        print(video_name)
        crop_and_align_images(join(images_path, video_name),
                       join(processed_images_path, video_name), landmark_model_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--landmark_model_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='Deepfakes')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c40')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            crop_and_align_all(**vars(args))
    else:
        crop_and_align_all(**vars(args))

    notify = Notify(endpoint="https://notify.run/Dbnkja3hR3rG7MuV")
    notify.send(" ".join(sys.argv) + " done")
