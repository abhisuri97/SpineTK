from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog

import sys
from split import make_split
from visualize import visualize_sample
import train 
from inference import inference, eval_img
import setup_catalogs


setup_logger()

baseline_directory = sys.argv[1] # path to directory
X_train, X_val, X_test, y_train, y_val, y_test = make_split(baseline_directory)
version = 'v1'
vert_metadata = setup_catalogs(version, 
               X_train, X_val, X_test,
               kpnames=['bottom_left', 'bottom_middle', 'bottom_right',
                        'top_right', 'top_middle', 'top_left'])
if sys.argv[2] == 'vis':
  visualize_sample(DatasetCatalog.get(f'vert{version}_train'), vert_metadata)

train(version, 1300, True)
predictor = inference("output/model_final.pth", 0.6)
eval_img(vert_metadata, "YOUR IMAGE PATH HERE.png", "out.png")