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
        try:
            img = read_image(os.path.join(root_dir, patient, "CT1.nii"), no_meta=True)
            print(patient, img.shape)
            shapes[patient] = img.shape
        except:
            print(patient, "failed")
    return shapes

def get_shapes_cbct(root_dir):
    shapes = {}

    for patient in os.listdir(root_dir):
        images = glob.glob(os.path.join(root_dir, patient, "X*.nii"))
        for cbct in images:
            try:
                m = re.search("X[0-9]+", cbct)
                n = m.group(0)
                img = read_image(cbct, no_meta=True)
                segmentations = glob.glob(os.path.join(root_dir, patient, "*_{}.nii".format(n.lower())))
                print(patient + "\\" + n, img.shape, len(segmentations))
                if len(segmentations) > 0:
                    shapes[patient + "\\" + n] = img.shape
            except: 
                print(patient, "failed")
    return shapes

def main(args):
    shapes = get_shapes_cbct(args.root_dir)
    print("Save shapes")
    pickle.dump(shapes, open(args.filename, 'wb'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data shapes')

    parser.add_argument("-root_dir", required=False, help="Get root directory of data", default="/data/cervix/patients")
    parser.add_argument("-filename", required=False, help="give filename for shapes pickle", default="CBCT_shapes.p")

    args = parser.parse_args()

    print(args.root_dir, args.filename)

    main(args)


