'''
Author: Tessa Wagenaar

Calculate the data statistics
'''
import glob
import os
from pathlib import Path
from utils.image_readers import read_image
import pickle
import numpy as np


files_CBCT = pickle.load(open("files_CBCT.p", 'rb'))
statistics = {}

print("calculate image data")
for (patient, shape, image_fn, segmentations) in files_CBCT:
    image, metadata = read_image(str(image_fn))
    print(f"Min: {image.min()} Max: {image.max()}")


pickle.dump(statistics, open("CBCT_statistics.p", 'wb'))
