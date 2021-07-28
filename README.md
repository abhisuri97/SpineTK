# SpineTK [WIP]

![](https://i.imgur.com/3nQo0jB.png)

This is a repository that contains the code to train a network for doing MR, CT, and X-ray image annotation (landmark annotation of six keypoints on individual vertebral bodies for vertebral height measurement). It is recommended that you run this on Google Colab. 

### Set up images/annotations

Images must be contained in a single folder with the following structure:

```
.
└── baseline-directory/
    ├── image-id-1/
    │   ├── augs/
    │   │   ├── 1/
    │   │   │   ├── aug-1.json
    │   │   │   └── aug-1.png
    │   │   ├── 2/
    │   │   │   ├── aug-2.json
    │   │   │   └── aug-2.png
    │   │   └── ...
    │   └── src/
    │       └── 0/
    │           ├── src-0.json
    │           └── src-0.png
    ├── image-id-2/
    │   ├── augs/
    │   │   ├── 1/
    │   │   │   ├── aug-1.json
    │   │   │   └── aug-1.png
    │   │   ├── 2/
    │   │   │   ├── aug-2.json
    │   │   │   └── aug-2.png
    │   │   └── ...
    │   └── src/
    │       └── 0/
    │           ├── src-0.json
    │           └── src-0.png
    ├── ...
    └── metadata.json
```

Each case gets its own folder. Under each folder, a case has image augmentations and a source image. The source image is a .png image with values in the range of 0-255. It is up to you how to get DICOM image values into this range; however, it has been useful for me to follow this rough algorithm: 

1. Pick only the most central slice of your image series (or just the single image if a radiograph)
2. Find the mean and standard deviation of all non-zero pixel values.
3. Use np.clip to "clip" values between -2.5 SD and +2.5 SD of the mean. 
4. Rescale the min/max values to 0 to 255. 
5. Save the image

This is some example code that implements that: 

```py
def read_img_and_kps(img_entry):
    ds = pydicom.dcmread(img_entry['dicom_path'])
    img = np.array(Image.fromarray(ds.pixel_array))
    mean, std = img.mean(), img.std()
    cl1 = np.clip((img-mean)/std, -2.5, 2.5)
    cl1 = ((cl1 - cl1.min()) * (1/(cl1.max() - cl1.min()) * 255)).astype(np.uint8)
    cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    return cl1
```

Augmentations can be generated using the `imgaug` python library: I use the following augmentation formula (has evolved since the original paper).

```py
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        sometimes(iaa.Crop(percent=(0, 0.1))),
        sometimes(iaa.Affine(
            rotate=(-5, 5),
        )),
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Add((-10, 10)),
                iaa.Multiply((0.5, 1.5)),
                iaa.LinearContrast((0.5, 2.0)),
            ],
            random_order=True
        )
    ],
    random_order=True
)
```

From there, you will need to generate a COCO compliant annotation file. The following method was used to do so in our paper. Note this isn't meant to be reproduced, but just to provide a guideline for how to generate augmented images with keypoints.

```py
# params: `img_entry` which is a dict with a filename  
# and whether the image contains hardware (we will 
# selectively augment these to a higher extent than other images). 
# `fname` is the filename, `aug_seq` is the augmentation sequence above
def read_img_and_plot_kps(img_entry, fname, aug_seq):
    try:
        #gets image (and stores in cl1) and keypoint entries [[X,Y], [X,Y], ...]
        cl1, kps = read_img_and_kps(img_entry) 
        # resize image and keypoints to 800px wide
        scale_pct = 800/cl1.shape[1]
        cl1 = cv2.resize(cl1,
            (int(cl1.shape[1]*scale_pct), int(cl1.shape[0]*scale_pct))
        )
        kps = kps*scale_pct
        
        # put image keypoints into separate array for img aug to work on them
        img_aug_kps = []
        for pt in kps:
            img_aug_kps.append(Keypoint(x=int(round(pt[0])), y=int(round(pt[1]))))
        kpsoi = KeypointsOnImage(img_aug_kps, shape=cl1.shape)
        
        # Here we selectively augment any images that have hardware
        tot_augs = 10 if img_entry['hardware'] == False else 30
        aug_images = [cl1 for _ in range(tot_augs)]
        aug_kps = [kpsoi for _ in range(tot_augs)]
        seq = aug_seq
        image_augs, kpsoi_augs = seq(images=aug_images, keypoints=aug_kps)
        
        # Save to folder
        baseline_dir = f"./baseline-directory/{fname}"
        if (os.path.isdir(baseline_dir)):
            shutil.rmtree(baseline_dir)
        os.makedirs(baseline_dir)
        if (os.path.isdir(f"{baseline_dir}/augs") == False):
            os.mkdir(f'{baseline_dir}/augs')
        if (os.path.isdir(f"{baseline_dir}/src") == False):
            os.mkdir(f'{baseline_dir}/src')
        image_augs.insert(0, cl1)
        kpsoi_augs.insert(0, kpsoi)
        for ida, i in enumerate(image_augs):
            # Now generate the COCO dict
            if ida > 0:
                os.mkdir(f'{baseline_dir}/augs/{ida}')
                cv2.imwrite(f'{baseline_dir}/augs/{ida}/aug-{ida}.png', i)
            else:
                os.mkdir(f'{baseline_dir}/src/{ida}')
                cv2.imwrite(f'{baseline_dir}/src/{ida}/src-{ida}.png', i)
            
            polygons = [
                list(
                    map(lambda p: [float(round(p.x)), float(round(p.y)), 2],
                        kpsoi_augs[ida][i: i + 4])
                )
                for i in range(0, len(kpsoi_augs[ida]), 4)
            ]
            
            bboxes = [get_bounding_box(p) for p in polygons]
            boxmode = 0 # BoxMode.XYXY_ABS from Detectron2 Docs
            
            writefile = f'{baseline_dir}/augs/{ida}/aug-{ida}' if ida > 0 else\
                        f'{baseline_dir}/src/{ida}/src-{ida}'
            with open(f"{writefile}.json", 'w') as outfile:
                obj = {
                    'file_name': f'{writefile}.png',
                    'height': int(i.shape[0]),
                    'width': int(i.shape[1]),
                    'image_id': f"{fname}-{'aug' if ida > 0 else 'src'}-{ida}",
                    'annotations': [
                        {'bbox': bboxes[idx], 
                         'bbox_mode': boxmode,
                         'category_id': 0,
                         'keypoints': list(itertools.chain(*p))
                        } for idx,p in enumerate(polygons)
                    ]
                }
                json.dump(obj, outfile)
        
        print(f"DONE WITH {fname}")
    except:
        print(f"COULD NOT DO {fname}")
        traceback.print_exc()
read_img_and_plot_kps(test_img, 'test', seq)

```


The .json files must have the following general format: 

```json
{
    'file_name': path to file,
    'height': image height in px,
    'width': image width in px,
    'image_id': unique name for the image (filepath will do),
    'annotations': [
        // for each of the vertebrae
        { 
            'bbox': bounding box array specified as 
                    [topleftX, topleftY, bottomRightX, bottomRightY],
            'bbox_mode': 0,
            'category_id': 0,
            'keypoints': [X1, Y1, 2.0, X2, Y2, 2.0, X3, Y3, 2.0, ...]
        }, 
    ]
}
```

You'll also need to generate a `metadata.json` file. The metadata file should contain the following: 

```json
{
    "image_id_that_correlated_to_folder_names": {
        "stratifying variable": True or False // (e.g. "Hardware": True)
    }, 
    ...
}
```

### Running [DOCUMENTATION Work in Progress!]

In a colab notebook, upload the code to the top level directory (files should be on the same level as the directory). You should also upload your images to colab. Then run these blocks of code (or run `run.py` after running `./install.sh`: 

```
./install.sh
```

```py
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog

import sys
from split import make_split
from visualize import visualize_sample
import train 
from inference import inference, eval_img
import setup_catalogs

setup_logger()
```

```py
baseline_directory = # path to directory with your images
X_train, X_val, X_test, y_train, y_val, y_test = make_split(baseline_directory)
version = 'v1' # change each time you run
vert_metadata = setup_catalogs(version, 
               X_train, X_val, X_test,
               kpnames=['bottom_left', 'bottom_middle', 'bottom_right',
                        'top_right', 'top_middle', 'top_left'])

visualize_sample(DatasetCatalog.get(f'vert{version}_train'), vert_metadata)
```

`train` takes in a version name, the number of iterations to train (to crudely implement early stopping, increase this 100 iters at a time and see if val_loss decreases by 10%. Note that the "True" argument means that the training will continue where you stopped. So you can run with train(version, 100, True) to train for 100 total epochs and then call train(version, 200, True) to train for 200 total epochs (trains the "100" model for 100 more epochs). 

```py
train(version, 1300, True)
```

Instantiate the predictor with a threshold confidence level. 0.6 is usually good to filter out the "bad" results with vertebrae that are too small.

```py
predictor = inference("output/model_final.pth", 0.6)
```

See the results! 2nd argument = path to the image, 3rd argument is the output path.

```py
eval_img(vert_metadata, "YOUR IMAGE PATH HERE.png", "out.png")
```
