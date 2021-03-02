# Yolov5
-The latest YOLO here being used for Traffic Signs recognition and classification on GTSRB and DFG datasets. The GTSRB(German Traffic Signs which consists of 43 classes with 30k   images) dataset is preprocessed as per the requirement(one text file per image etc.) and then fed to the model. Also, the second dataset called DFG Traffic Sign Data Set(Slovenian traffic sign dataset) is heavier(have more resolution) than GTSRB though the number of images are much lesser.

GTSRB-> https://benchmark.ini.rub.de/gtsrb_news.html
DFG-> https://www.vicos.si/Downloads/DFGTSD

-The accuracy achieved so far is 95% precision and 99% recall.The different results logs, PR curve, F1 score etc. can be seen under Wandb folder and 'runs/train' and 'runs/detect'.

-The model is being tuned to work on the images taken from the onboard camera of the Autonomous vehicle where it recognizes and classifies several traffic signs in the scenario , captured by the camera attached to the vehicle.

The different techniques like below are in progress:
-Data Augmentation 
-Prevention of Overfitting and saving the last best weights
-Multiple GPUs (DDP from PyTorch)
-Exploration of best learning rates
-Feasibilty of Ensembe of Models
-Different Losses like Classification Loss, Object Loss and Box loss
-The status of the weights being currently used in evry epoch
-Different Image sizes
-Usage of Wandb and Tesnorboard for better visualization of logs and results
-Using Inference on different images from the surroundings and getting better confidence score


  
