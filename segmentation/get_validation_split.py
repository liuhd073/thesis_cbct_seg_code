import pickle

image_shapes_CT = pickle.load(open("CT_shapes.p", 'rb'))
image_shapes_CBCT = pickle.load(open("CBCT_shapes.p", 'rb'))

patients = list(image_shapes_CT.keys())
patients_CBCT = list(image_shapes_CBCT.keys())

patients_CT_train = patients[:int(0.8*len(patients))]
patients_CT_val = patients[int(0.8*len(patients)):]
patients_CBCT_train = [p for p in patients_CBCT if p.split('\\')[0] in patients_CT_train]
patients_CBCT_val = [p for p in patients_CBCT if p.split('\\')[0] in patients_CT_val]

image_shapes_CT_train = {p: image_shapes_CT[p] for p in patients_CT_train}
image_shapes_CT_val = {p: image_shapes_CT[p] for p in patients_CT_val}
image_shapes_CBCT_train = {p: image_shapes_CBCT[p] for p in patients_CBCT_train}
image_shapes_CBCT_val = {p: image_shapes_CBCT[p] for p in patients_CBCT_val}

pickle.dump(image_shapes_CT_train, open("CT_shape_train.p", 'wb'))
pickle.dump(image_shapes_CT_val, open("CT_shape_validation.p", 'wb'))
pickle.dump(image_shapes_CBCT_train, open("CBCT_shape_train.p", 'wb'))
pickle.dump(image_shapes_CBCT_val, open("CBCT_shape_validation.p", 'wb'))
