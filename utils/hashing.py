#hashing.py

import pandas as pd 
import imagehash
from PIL import Image
from tqdm import tqdm

import os 

directory = '../data/'

def get_duplicated_images(dir_path,hash_func= imagehash.phash):
    result = []
    for directory in os.listdir(dir_path):
        images = {}
        duplicated_images = []
        for image in sorted(os.listdir(dir_path + directory)):
            path = dir_path + directory + '/' + image

            try:
                hashing = hash_func(Image.open(path))
            except:
                print('IMAGE ERROR :'.join(path)+'\n')

            if hashing in images:
                print('IMAGE DUPLICATED AS :'.join(images[hashing])+'\n')
                duplicated_images.append(images[hashing])

            images[hashing] = images.get(hashing,[]) + [path]

        result.append(duplicated_images)

    return result

get_duplicated_images(directory)
