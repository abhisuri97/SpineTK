from detectron2.data import MetadataCatalog, DatasetCatalog
import return_dicts

def setup_catalogs(version, train, val, test, kpnames):
  for d in ["train", "val", "test"]:
      DatasetCatalog.register(f"vert{version}_" + d, 
        lambda d=d: return_dicts(d, train, val, test)
      )
      MetadataCatalog.get(f"vert{version}_" + d).set(thing_classes=["vertebrae"],
                                          keypoint_names=kpnames,
                                          keypoint_flip_map=[])
  vert_train_metadata = MetadataCatalog.get(f"vert{version}_train")
  return vert_train_metadata