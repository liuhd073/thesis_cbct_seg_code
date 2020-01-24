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


root_dir = Path("/data/cervix/patients")

statistics = {}

# List all the patients
statistics["patients"] = os.listdir(root_dir)
statistics["n_patients"] = len(statistics["patients"])


MIN_HV = -100
MAX_HV = 300

print("calculate image data")
for patient in statistics["patients"]:
    print(patient)
    images = glob.glob(str(root_dir / patient / "*.nii"))
    for img in images:
        if ("cervix_uterus_" not in img) and ("bladder_" not in img):
            if img not in statistics:
                statistics[img] = {}
            image, metadata = read_image(img)
            hist, bins = np.histogram(image, bins=100, range=(MIN_HV, MAX_HV))
            statistics[img]["image_hist"] = (hist, bins)
            statistics[img]["min"] = image.min()
            statistics[img]["max"] = image.max()
            statistics[img]["mean"] = image.mean()
            clipped_image = np.clip(image, MIN_HV, MAX_HV)
            normalized_image = (clipped_image - MIN_HV) / (MAX_HV - MIN_HV)
            statistics[img]["norm_mean"] = (normalized_image).mean()
            hist_norm, bins_norm = np.histogram(
                normalized_image, bins=100, range=(0, 1))
            statistics[img]["norm_image_hist"] = (hist_norm, bins_norm)
            print(hist)
print(bins, bins_norm)

pickle.dump(statistics, open("image_statistics.p", 'wb'))


# Get Percentages of images contain segmentations
