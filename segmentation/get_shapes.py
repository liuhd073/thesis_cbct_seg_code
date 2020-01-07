import os
import re
import glob
import pickle
import argparse
from utils.image_readers import read_image


# root_dir = "D:\data\cervix\patients"
# filename = "CBCT_shapes.p"

def get_shapes(root_dir):
    shapes = {}
    for patient in os.listdir(root_dir):
        img = read_image(os.path.join(root_dir, patient, "CT1.nii"), no_meta=True)
        print(img.shape)
        shapes[patient] = img.shape

    return shapes

def get_shapes_cbct(root_dir):
    shapes = {}

    for patient in os.listdir(root_dir):
        images = glob.glob(os.path.join(root_dir, patient, "X*.nii"))
        for cbct in images:
            m = re.search("X[0-9]+", cbct)
            n = m.group(0)
            img = read_image(cbct, no_meta=True)
            segmentations = glob.glob(os.path.join(root_dir, patient, "*_{}.nii".format(n.lower())))
            print(patient + "\\" + n, img.shape, len(segmentations))
            if len(segmentations) > 0:
                shapes[patient + "\\" + n] = img.shape
    return shapes

def main(args):
    shapes = get_shapes_cbct(args.root_dir)
    pickle.dump(shapes, open(args.filename, 'wb'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("root_dir", help="Get root directory of data")
    parser.add_argument("filename", type=str, help="give filename for shapes pickle")

    args = parser.parse_args()

    main(args)


