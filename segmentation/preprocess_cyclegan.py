"""
Author: Tessa Wagenaar
"""

from pathlib import Path 
from utils.image_readers import read_image
from utils.image_writers import write_image
from skimage import measure
from scipy import stats
import skimage
import numpy as np
from remove_table import get_table
from preprocess import *
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def process_CT(source_dir, target_dir):
    files = list(source_dir.rglob("CT.nrrd"))
    train = files[:int(0.8*len(files))]
    val = files[int(0.8*len(files)):int(0.9*len(files))]
    test = files[int(0.9*len(files)):]
    transform_CT= transforms.Compose(
        [ClipAndNormalize(250, 1750)])

    image_index = 0
    # A: CT, B: CBCT
    for i, f in enumerate(tqdm(train)):
        image, meta = read_image(str(f))

        bboxes = np.array(get_table(image, clip_val=300, margin=2500, show_imgs=False))
        if len(bboxes) == 0:
            print("Table not found in image!")
            image_index = 0
            image = transform_CT({"image": image})["image"]
            for X in image[10:-10]:
                image_index += 1
                im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
                im.save(f"TEMP/NO_TABLE_{image_index}_{f.parent.stem}.jpg")

        bbox = (stats.mode(bboxes[:,0])[0][0], stats.mode(bboxes[:,1])[0][0], stats.mode(bboxes[:,2])[0][0], stats.mode(bboxes[:,3])[0][0])
        image = np.clip(image, 0, image.max())
        image[:,bbox[0]:512, bbox[1]:bbox[3]] = 0

        image = transform_CT({"image": image})["image"]
        for X in image[10:-10]:
            image_index += 1
            im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
            im.save(str(target_dir / "trainA" / f"{image_index}_{f.parent.stem}.jpg"))
        if temp_save:
            im.save(f"TEMP/TRAIN_A_{image_index}_{f.parent.stem}.jpg")
    
    image_index = 0
    for f in tqdm(val):
        image, meta = read_image(str(f))

        bboxes = np.array(get_table(image, clip_val=200, margin=2500, show_imgs=False))
        if len(bboxes) == 0:
            print("Table not found in image!")

        bbox = (stats.mode(bboxes[:,0])[0][0], stats.mode(bboxes[:,1])[0][0], stats.mode(bboxes[:,2])[0][0], stats.mode(bboxes[:,3])[0][0])
        image = np.clip(image, 0, image.max())
        image[:,bbox[0]:512, bbox[1]:bbox[3]] = 0

        image = transform_CT({"image": image})["image"]
        for X in image[10:-10]:
            image_index += 1
            im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
            im.save(str(target_dir / "valA" / f"{image_index}_{f.parent.stem}.jpg"))
        if temp_save:
            im.save(f"TEMP/VAL_A_{image_index}_{f.parent.stem}.jpg")

    image_index = 0
    for f in tqdm(test):
        image, meta = read_image(str(f))

        bboxes = np.array(get_table(image, clip_val=200, margin=2500, show_imgs=False))
        if len(bboxes) == 0:
            print("Table not found in image!")
            image_index = 0
            image = transform_CT({"image": image})["image"]
            for X in image[10:-10]:
                image_index += 1
                im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
                im.save(f"TEMP/NO_TABLE_{image_index}_{f.parent.stem}.jpg")

        bbox = (stats.mode(bboxes[:,0])[0][0], stats.mode(bboxes[:,1])[0][0], stats.mode(bboxes[:,2])[0][0], stats.mode(bboxes[:,3])[0][0])
        image = np.clip(image, 0, image.max())
        image[:,bbox[0]:512, bbox[1]:bbox[3]] = 0

        image = transform_CT({"image": image})["image"]
        for X in image[10:-10]:
            image_index += 1
            im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
            im.save(str(target_dir / "testA" / f"{image_index}_{f.parent.stem}.jpg"))
        if temp_save:
            im.save(f"TEMP/TEST_A_{image_index}_{f.parent.stem}.jpg")


def process_CBCT(source_dir, target_dir):
    files = list(source_dir.iterdir())
    files = [f for f in files if len(list(f.glob("X*.nrrd"))) > 0]

    train = files[:int(0.8*len(files))]
    val = files[int(0.8*len(files)):int(0.9*len(files))]
    test = files[int(0.9*len(files)):]
    transform_CT= transforms.Compose(
        [ClipAndNormalize(250, 1750)])

    scan_id = 0
    image_index = 0
    for p in tqdm(train):
        cbct_count = 1
        for f in p.glob("X*.nrrd"):
            if cbct_count > 2: break
            image, meta = read_image(str(f))
            cbct_count += 1
            image = transform_CT({"image": image})["image"]
            scan_id += 1
            for X in image[10:-10]:
                image_index += 1
                im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
                im.save(str(target_dir / "trainB" / f"{image_index}_{f.parent.stem}_{f.stem}.jpg"))
            if temp_save:
                im.save(f"TEMP/TRAIN_B_{image_index}_{f.parent.stem}_{f.stem}_{scan_id}.jpg")

    image_index = 0
    for p in tqdm(val):
        cbct_count = 1
        for f in p.glob("X*.nrrd"):
            if cbct_count > 2: break
            if f.stem == "X01":
                continue
            image, meta = read_image(str(f))
            cbct_count += 1
            image = transform_CT({"image": image})["image"]
            for X in image[10:-10]:
                image_index += 1
                im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
                im.save(str(target_dir / "valB" / f"{image_index}_{f.parent.stem}_{f.stem}.jpg"))
            if temp_save:
                im.save(f"TEMP/VAL_B_{image_index}_{f.parent.stem}_{f.stem}.jpg")

    image_index = 0
    scan_id = 0
    # A: CT, B: CBCT
    for p in tqdm(test):
        cbct_count = 1
        for f in p.glob("X*.nrrd"):
            if cbct_count > 2: break
            cbct_count += 1
            scan_id += 1
            image, meta = read_image(str(f))
            image = transform_CT({"image": image})["image"]
            for X in image[10:-10]:
                image_index += 1
                im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
                im.save(str(target_dir / "testB" / f"{image_index}_{f.parent.stem}_{f.stem}.jpg"))
            if temp_save:
                im.save(f"TEMP/TEST_B_{image_index}_{f.parent.stem}_{f.stem}_{scan_id}.jpg")


if __name__ == "__main__":
    CT_folder = Path("/data/Cervix_COMPLETE")
    CBCT_folder = Path("/data/cervix/CBCT")
    dest = Path("/data/cyclegan/cbct")

    temp_save = False

    process_CT(CT_folder, dest)
    process_CBCT(CBCT_folder, dest)

