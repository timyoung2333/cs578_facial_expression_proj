#!/usr/bin/env python3
# The images were provided in the csv files.
# This script converts them to jpeg files.
import sys
import cv2
import numpy as np
from tqdm import tqdm

filename = "icml_face_data.csv"
save_dir = "images"

def save_image(line, img_name):

    emotion, usage, pixels = line.split(",")
    pixels = [int(i) for i in pixels.split(" ")]
    pixels = np.asarray(pixels).reshape((48, 48))
    cv2.imwrite(img_name, pixels)

def main():

    with open(filename) as f:

        counter = 0
        lines = f.readlines()
        for line in tqdm(lines[1:]):
            save_image(line, "{}/{:05d}.jpg".format(save_dir, counter))
            counter += 1


if __name__ == "__main__":
    main()
