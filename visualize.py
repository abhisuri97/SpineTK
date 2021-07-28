import random
import cv2
from detectron2.utils.visualizer import Visualizer

def visualize_sample(ds_catalog, metadata):
  count = 0
  for d in random.sample(ds_catalog, 2):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite(out.get_image()[:, :, ::-1], f'{count}.png')
    count += 1