#image_formatter.py

from PIL import Image
from tqdm import tqdm

import os

directory = '../data/'

def image_formatter(dir_path):
    for directory in os.listdir(dir_path):
        os.makedirs(os.path.dirname('../format/'+directory+'/'), exist_ok=True)
        counter = 0
        for image in tqdm(os.listdir(dir_path + directory)):
            path = dir_path + directory + '/' + image
            try:
                img = Image.open(path)
                img = img.convert('RGB')
                img.save('../format/'+directory+'/'+directory+'_'+str(counter)+'.jpg','JPEG')
                counter += 1
            except:
                continue
        print(directory+': DONE')
            
image_formatter(directory)