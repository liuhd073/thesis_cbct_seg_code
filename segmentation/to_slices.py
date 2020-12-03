"""
Author: Tessa Wagenaar
"""

from preprocess import ClipAndNormalize
from utils.plotting import plot_2d
from utils.image_readers import read_image
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image
from pathlib import Path

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main_CT():
    target_dir = Path("/data/cyclegan/cbct")
    files = list(Path("/data/cervix/patients").iterdir())
    files = [f for f in files if len(list(f.iterdir())) > 0]

    transform_CT= transforms.Compose(
        [ClipAndNormalize(250, 1750)])

    j=0
    image_index = 0

    for p in tqdm(files):
        for f in p.glob("CT*.nii"):
            image, meta = read_image(str(f))
            image = transform_CT({"image": image})["image"]

            for X in image:
                image_index += 1
                im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
                im.save(str(target_dir / "test_CT" / f"{image_index}_{f.parent.stem}_{f.stem}.jpg"))
            if temp_save:
                im.save(f"TEMP/{image_index}_{f.parent.stem}_{f.stem}.jpg")



def main_CBCT():
    target_dir = Path("/data/cyclegan/cbct")
    files = list(Path("/data/cervix/patients").iterdir())
    files = [f for f in files if len(list(f.iterdir())) > 0]

    transform_CT= transforms.Compose(
        [ClipAndNormalize(250, 1750)])

    j=0
    image_index = 0

    for p in tqdm(files):
        for f in p.glob("X*.nii"):
            image, meta = read_image(str(f), ref_fn=(p / "CT1.nii"), affine_matrix=True)
            image = transform_CT({"image": image})["image"]

            for X in image:
                image_index += 1
                im = Image.fromarray(np.uint8(X.squeeze() * 255), 'L')
                im.save(str(target_dir / "test_CBCT" / f"{image_index}_{f.parent.stem}_{f.stem}.jpg"))
            if temp_save:
                im.save(f"TEMP/{image_index}_{f.parent.stem}_{f.stem}.jpg")


if __name__ == "__main__":
    temp_save = False
    main_CT()
    main_CBCT()
