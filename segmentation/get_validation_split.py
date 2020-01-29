import pickle
import argparse


def main(args):
    print("CT input:", args.CT_filename)
    print("CT output:", args.output_CT_filename + "_train.p")
    image_shapes_CT = pickle.load(open(args.CT_filename, 'rb'))
    patients = list(image_shapes_CT.keys())

    patients_CT_train = patients[:int(args.ratio*len(patients))]
    patients_CT_val = patients[int(args.ratio*len(patients)):]
    print(patients_CT_val)

    image_shapes_CT_train = {p: image_shapes_CT[p] for p in patients_CT_train}
    image_shapes_CT_val = {p: image_shapes_CT[p] for p in patients_CT_val}

    pickle.dump(image_shapes_CT_train, open(
        args.output_CT_filename + "_train.p", 'wb'))
    pickle.dump(image_shapes_CT_val, open(
        args.output_CT_filename + "_validation.p", 'wb'))

    if args.CBCT:
        image_shapes_CBCT = pickle.load(open("CBCT_shapes.p", 'rb'))
        patients_CBCT = list(image_shapes_CBCT.keys())

        patients_CBCT_train = [p for p in patients_CBCT if p.split('\\')[
            0] in patients_CT_train]
        patients_CBCT_val = [p for p in patients_CBCT if p.split('\\')[
            0] in patients_CT_val]

        image_shapes_CBCT_train = {
            p: image_shapes_CBCT[p] for p in patients_CBCT_train}
        image_shapes_CBCT_val = {
            p: image_shapes_CBCT[p] for p in patients_CBCT_val}

        pickle.dump(image_shapes_CBCT_train, open(
            args.output_CBCT_filename + "_train.p", 'wb'))
        pickle.dump(image_shapes_CBCT_val, open(
            args.output_CBCT_filename + "_validation.p", 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("CT_filename", help="Give shapes filename")
    parser.add_argument("-CBCT_filename", required=False)

    parser.add_argument("-output_CT_filename",
                        required=False, default="CT_shapes")
    parser.add_argument("-output_CBCT_filename",
                        required=False, default="CBCT_shapes")

    parser.add_argument("-CBCT", required=False,
                        default=False, action="store_true")
    parser.add_argument("-ratio", required=False, default=0.8)

    args = parser.parse_args()

    if args.CBCT and (args.CBCT_filename is None or args.output_CBCT_filename is None):
        parser.error(
            "input and output file for CBCT need to be given in case of -CBCT")

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
