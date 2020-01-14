import pickle

image_shapes_CT = pickle.load(open("CT_shapes.p", 'rb'))
image_shapes_CBCT = pickle.load(open("CBCT_shapes.p", 'rb'))

patients = list(image_shapes_CT.keys())

patients_CT_train = patients[:int(0.8*len(patients))]
patients_CT_val = patients[int(0.8*len(patients)):]
patients_CBCT_train = patients[:int(0.8*len(patients))]
patients_CBCT_val = patients[int(0.8*len(patients)):]
print(len(patients_CT_train), len(patients))

pickle.dump(patients_CT_train, open("CT_shape_train.p", 'wb'))
pickle.dump(patients_CT_val, open("CT_shape_validation.p", 'wb'))
pickle.dump(patients_CBCT_train, open("CBCT_shape_train.p", 'wb'))
pickle.dump(patients_CBCT_val, open("CBCT_shape_validation.p", 'wb'))
