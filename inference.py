from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import os
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

def inference(version, filepath, thresh):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TEST = (f"vert{version}_val",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.STEPS = []        # do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vertebrae)
  cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 6

  cfg.MODEL.WEIGHTS = os.path.join(filepath)  # path to the model we just trained
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh   # set a custom testing threshold
  cfg.TEST.KEYPOINT_OKS_SIGMAS = [1,1,1,1]
  predictor = DefaultPredictor(cfg)
  return predictor

def eval_img(metadata, predictor, img_path, output_path):
      im = cv2.imread(img_path)
      outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
      for o in outputs['instances'].get_fields()['pred_keypoints']:
        for k in o:
          k[2] = 2
      scale = 1

      v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=scale, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
      )
      # Implement logic here for writing out keypoints
      cv2.imwrite(v, output_path)