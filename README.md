# Yolov5
-The latest YOLO here being used for Traffic Signs recognition and classification on GTSRB and DFG datasets. The GTSRB(German Traffic Signs which consists of 43 classes with 30k   images) dataset is preprocessed as per the requirement(one text file per image etc.) and then fed to the model. Also, the second dataset called DFG Traffic Sign Data Set(Slovenian traffic sign dataset) is heavier(have more resolution) than GTSRB though the number of images are much lesser.

GTSRB-> https://benchmark.ini.rub.de/gtsrb_news.html
DFG-> https://www.vicos.si/Downloads/DFGTSD

-The accuracy achieved so far is 95% precision and 99% recall.The different results logs, PR curve, F1 score etc. can be seen under Wandb folder and 'runs/train' and 'runs/detect'.

-The model is being tuned to work on the images taken from the onboard camera of the Autonomous vehicle where it recognizes and classifies several traffic signs in the scenario , captured by the camera attached to the vehicle.

The different techniques like below are in progress:
- Data Augmentation 
- Prevention of Overfitting and saving the last best weights
- Multiple GPUs (DDP from PyTorch)
- Exploration of best learning rates
- Feasibilty of Ensembe of Models
- Different Losses like Classification Loss, Object Loss and Box loss
- The status of the weights being currently used in evry epoch
- Freezing of some layers for Transfer Learning
- Different Image sizes
- Usage of Wandb and Tesnorboard for better visualization of logs and results
- Using Inference on different images from the surroundings and getting better confidence score

The libraries being used here are: PyTorch, matplotlib, numpy, opencv, tensorboard, scikit-learn, wandb


## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

## Inference

detect.py runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on example images in `data/images`:
```bash
$ python detect.py --source data/images --weights yolov5s.pt --conf 0.25

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', project='runs/detect', save_conf=False, save_txt=False, source='data/images/', update=False, view_img=False, weights=['yolov5s.pt'])
YOLOv5 v4.0-96-g83dc1b4 torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16160.5MB)

Fusing layers... 
Model Summary: 224 layers, 7266973 parameters, 0 gradients, 17.0 GFLOPS
image 1/2 /content/yolov5/data/images/bus.jpg: 640x480 4 persons, 1 bus, Done. (0.010s)
image 2/2 /content/yolov5/data/images/zidane.jpg: 384x640 2 persons, 1 tie, Done. (0.011s)
Results saved to runs/detect/exp2
Done. (0.103s)
```
<img src="https://user-images.githubusercontent.com/26833433/97107365-685a8d80-16c7-11eb-8c2e-83aac701d8b9.jpeg" width="500">  


## Training

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)
https://www.ultralytics.com.

