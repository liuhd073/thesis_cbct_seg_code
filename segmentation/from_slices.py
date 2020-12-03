"""
Author: Tessa Wagenaar
"""

from datasets.dataset import CTDataset
from datasets.dataset_CBCT import CBCTDataset
from datasets.dataset_combined import CombinedDataset
from datasets.dataset_duo import DuoDataset
from preprocess import *
from utils.plotting import plot_2d
from torch.utils.data import DataLoader
from utils.image_readers import read_image
from utils.image_writers import write_image

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import defaultdict

import re
import pickle
import argparse
import torch
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def sort_slices(args):
    CBCT_slices = {"real": [], "fake": []}
    for i, file in enumerate(args.input_dir.iterdir()):
        match = re.match(r".*[0-9]*\_[0-9]*\_*[0-9]+.*\_real\.png", str(file))
        if match: 
            CBCT_slices["real"].append(match[0])
        match = re.match(r".*[0-9]+.*\_fake\.png", str(file))
        if match: 
            CBCT_slices["fake"].append(match[0])


    slices = {"real": defaultdict(lambda: defaultdict(list)), "fake": defaultdict(lambda: defaultdict(list))}
    for s in CBCT_slices["real"]:
        patient, scan = s.split("_")[1], s.split("_")[2]
        slices["real"][patient][scan].append((int(s.split("_")[0].split("/")[1]), s))
    for s in CBCT_slices["fake"]:
        patient, scan = s.split("_")[1], s.split("_")[2]
        slices["fake"][patient][scan].append((int(s.split("_")[0].split("/")[1]), s))

    return slices


def save_scans(args, slices):
    source_dir = Path("/data/cervix/patients")
    for patient, scans in tqdm(slices["fake"].items()):
        for scan, s in scans.items():
            s = sorted(s, key=lambda tup: tup[0])
            image = []
            for tup in s:

                im_frame = Image.open(tup[1]).convert('L')
                np_array = np.array(im_frame.getdata()).reshape((512,512)) / 255.0
                image.append(np_array)

            img = np.stack(image)
            img = img * 1500 + 250
            if not (source_dir / patient / (f"bladder_{scan.lower()}.nii")).exists():
                continue
            _, meta = read_image(str(source_dir / patient / ("CT1.nii")))
            bladder = read_image(str(source_dir / patient / (f"bladder_{scan.lower()}.nii")), no_meta=True, ref_fn=str(source_dir / patient / ("CT1.nii")), affine_matrix=True)
            cervix = read_image(str(source_dir / patient / (f"cervix_uterus_{scan.lower()}.nii")), no_meta=True, ref_fn=str(source_dir / patient / ("CT1.nii")), affine_matrix=True)

            if not (args.output_dir / patient).exists(): (args.output_dir / patient).mkdir()
            write_image(img, str(args.output_dir / patient / (scan + ".nrrd")), metadata=meta)
            write_image(bladder, str(args.output_dir / patient / (f"bladder_{scan.lower()}.nrrd")), metadata=meta)
            write_image(cervix, str(args.output_dir / patient / (f"cervix_uterus_{scan.lower()}.nrrd")), metadata=meta)


def main(args):
    slices = sort_slices(args)
    save_scans(args, slices)
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Create Scans from CycleGAN output images')

    parser.add_argument("-input_dir", help="Directionary with images to be converted",
                        default=None, required=True)
    parser.add_argument("-output_dir", help="Directionary to put converted scans",
                        default=None, required=True)

    args = parser.parse_args()
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
