from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
import os 

def train(version, iters, resume=True):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TRAIN = (f"vert{version}_train",)
  cfg.DATASETS.TEST = (f"vert{version}_val",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  cfg.SOLVER.MAX_ITER = iters    
  cfg.SOLVER.STEPS = []        # do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vertebrae)
  cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 6

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=resume)
  trainer.train()