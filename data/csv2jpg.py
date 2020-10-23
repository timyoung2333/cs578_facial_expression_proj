#!/usr/bin/env python3
# The images were provided in the csv files.
# This script converts them to jpeg files.
import sys
import cv2
import numpy as np

filename = "icml_face_data.csv"
save_dir = "images/"

def save_image(line, img_name):

    emotion, usage, pixels = line.split(",")
    pixels = [int(i) for i in pixels.split(" ")]
    print(len(pixels))
    exit(0)

def main():

    with open(filename) as f:

        counter = 0
        lines = f.readlines()
        for line in lines[1:]:
            save_image(line, "{:05d}.jpg".format(counter))
            counter += 1


if __name__ == "__main__":
    main()
