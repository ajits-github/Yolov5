# Yolov5
-The latest YOLO here being used for Traffic Signs recognition and classification on GTSRB and DFG datasets. The GTSRB(German Traffic Signs which consists of 43 classes with 30k   images) dataset is preprocessed as per the requirement(one text file per image etc.) and then fed to the model. Also, the second dataset called DFG Traffic Sign Data Set(Slovenian traffic sign dataset) is heavier(have more resolution) than GTSRB though the number of images are much lesser.

GTSRB-> https://benchmark.ini.rub.de/gtsrb_news.html
DFG-> https://www.vicos.si/Downloads/DFGTSD

-The accuracy achieved so far is 95% precision and 99% recall.The different results logs, PR curve, F1 score etc. can be seen under Wandb folder and 'runs/train' and 'runs/detect'.

-The model is being tuned to work on the images taken from the onboard camera of the Autonomous vehicle where it recognizes and classifies several traffic signs in the scenario , captured by the camera attached to the vehicle.

The different techniques like below are in progress:
- Data Augmentation 
- Freezing of some layers for Transfer Learning from DFG and GTSRB to GTSDB. Retraining by unfreezing the whole network but with different learning rates for backbone and head.
- Prevention of Overfitting (Early Stopping) and saving the last best weights
- Multiple GPUs (DDP from PyTorch)
- Exploration of best learning rates
- Feasibilty of Ensembe of Models (Inferences using the combination of YOLOv3 and YOLOv5)
- Different Losses like Classification Loss, Object Loss and Box loss
- The status of the weights being currently used in evry epoch
- Different Image sizes (Multi-scale training)
- Usage of Wandb and Tesnorboard for better visualization of logs and results
- Using Inference on different images from the surroundings and getting better confidence score
- For results, look at yolov5/runs/train/exp58 (GTSDB) , yolov5/runs/train/exp26 (DFG) and yolov5/runs/train/exp14 (GTSRB)
The libraries being used here are: PyTorch, matplotlib, numpy, opencv, tensorboard, scikit-learn, wandb


## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

## Inference

```

To run inference on example images in `data/images`:
```bash
$ python detect.py --source data/images --weights yolov5s.pt --conf 0.25



## Training

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```



## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)
https://www.ultralytics.com.

