import numpy as np

MIN_HV = -100
MAX_HV = 300

class Clip(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return np.clip(image, MIN_HV, MAX_HV)

class NormalizeHV(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return (image - MIN_HV) / (MAX_HV - MIN_HV)

class NormalizeIMG(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())