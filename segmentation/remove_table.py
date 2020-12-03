from pathlib import Path 
from utils.image_readers import read_image
from utils.image_writers import write_image
from skimage import measure
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def get_table(image, clip_val=300, margin=2500, show_imgs=False):
    bboxes = []
    image = np.clip(image, 0, image.max())
    for i, im in enumerate(image):
        im_white = np.clip(im, clip_val, clip_val+1) - clip_val

        im = Image.fromarray(np.uint8(im_white * 255), 'L')

        labels_mask = measure.label(im_white)
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            if ((regions[1].area + margin) > regions[0].area):
                start_id = 2
                labels_mask[regions[1].coords[:,0], regions[1].coords[:,1]] = 0
            elif (regions[1].centroid[1] < 150) or (regions[1].centroid[1] > 400):
                start_id = 2
                labels_mask[regions[1].coords[:,0], regions[1].coords[:,1]] = 0
                if len(regions) > 2:
                    if (regions[2].centroid[1] > 450):
                        start_id = 3
                        labels_mask[regions[2].coords[:,0], regions[2].coords[:,1]] = 0
            else:
                start_id = 1
            labels_mask[regions[0].coords[:,0], regions[0].coords[:,1]] = 0
        else: continue

        if regions[start_id].area < 1000 or regions[start_id].centroid[0] < 200:
            continue
        if len(regions) > start_id:
            for rg in regions[(start_id + 1):]:
                labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
        labels_mask[labels_mask!=0] = 1
        mask = labels_mask
        bbox = regions[start_id].bbox
        bboxes.append(bbox)

        if show_imgs:
            np_bboxes = np.array(bboxes)
            bbox = [np_bboxes[:,0].min(), np_bboxes[:,1].min(), np_bboxes[:,2].max(), np_bboxes[:,3].max()]
            fig, axes = plt.subplots(1,4, figsize=(12,4))
            axes[0].imshow(im, cmap="gray")
            axes[1].imshow(im_white, cmap="gray")
            axes[2].imshow(mask, cmap="gray")
            axes[3].imshow(im, cmap="gray")
            rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],linewidth=1,edgecolor='r',facecolor='none')
            axes[3].add_patch(rect)
            fig.show()
            plt.show()
    return bboxes    

if __name__ == "__main__":
    data = Path("/data/Cervix_COMPLETE")
    data_dest = Path("/data/cervix/no_table")

    for i, patient in enumerate(data.iterdir()):
        image, meta = read_image(str(patient / "CT.nrrd"))
        bboxes = np.array(get_table(image, clip_val=200, margin=2500, show_imgs=False))
        bbox = (bboxes[:,0].min(), bboxes[:,1].min(), bboxes[:,2].max(), bboxes[:,3].max())
        no_table = np.clip(image, 0, image.max())
        no_table[:,bbox[0]+10:512, bbox[1]:bbox[3]] = 0
        if not (data_dest / (patient.stem )).exists(): (data_dest / (patient.stem )).mkdir()
        write_image(no_table, str(data_dest / (patient.stem ) / "CT1.nrrd"), metadata=meta)
        