from detectron2.structures import BoxMode
import json
import os

def get_dicts(baseline_directory, img_list, tr_type="train"):
  # Description: Will take in a base directory containing all images w/augmentations
  # and return a list of dataset directionaries compatible with Detectron2 that are
  # in the img_list
  dataset_dicts = []
  for x in os.listdir(f"{baseline_directory}"):
    print(x)
    if (os.path.isdir(f"{baseline_directory}/{x}") and (int(x) in img_list)):
      # process src image. This is the non augmented one
      if (os.path.isdir(f"{baseline_directory}/{x}/src/0")):
        with open(f'{baseline_directory}/{x}/src/0/src-0.json') as json_file:
          d = json.load(json_file)
          d['bbox_mode'] = BoxMode.XYXY_ABS
          
          # double checking...
          throwout = False
          for a in d['annotations']:
            if len(a['keypoints']) != 18:
              throwout = True
          if throwout == False:
            dataset_dicts.append(d)
      else:
        print(f"ERROR on {x}")
        continue
      if (tr_type == "train" or tr_type == "val"):
        for a in os.listdir(f"{baseline_directory}/{x}/augs"):
          if (os.path.isdir(f"{baseline_directory}/{x}/augs/{a}")):
            with open(f'{baseline_directory}/{x}/augs/{a}/aug-{a}.json') as json_file:
              d = json.load(json_file)
              d['bbox_mode'] = BoxMode.XYXY_ABS
              throwout = False
              for a in d['annotations']:
                if len(a['keypoints']) != 18:
                  throwout = True
              if throwout == False:
                dataset_dicts.append(d)
              
  return dataset_dicts