import torch
import os

# Ensure that you have installed the YOLOv5 dependencies
# You can install them using: pip install -r requirements.txt
# Make sure to have your dataset directory structured according to YOLOv5 requirements

# Path to YOLOv5 repository
yolov5_path = 'C:/Users/RO100202/yolov5'

# Path to your train, val, and test dataset directories
train_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/train'
val_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/val'
test_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/LICENTA/dataset_final_15.04/test'

# Path to save trained weights
weights_path = 'C:/Users/RO100202/yolov5/runs/train/my_model/weights'

# Path to your data YAML file
data_yaml_path = 'C:/Users/RO100202/pythonProject/models/yolov5s.yaml'

# YOLOv5 model configuration
config = f'yolov5s.yaml'  # You can choose different model sizes: yolov5s, yolov5m, yolov5l, or yolov5x

# Training command for training on train dataset
train_command = f'python {yolov5_path}/train.py --batch 16 --epochs 60 --data {data_yaml_path} --cfg {yolov5_path}/models/{config} --weights yolov5s.pt --name my_model'

# Change directory to YOLOv5 repository
os.chdir(yolov5_path)

# Run training command on train dataset
#os.system(train_command)

# Evaluation command for evaluating on val dataset
eval_command = f'python {yolov5_path}/val.py --weights {weights_path}/last.pt --data {data_yaml_path}  --conf 0.5 --iou 0.65'

# Run evaluation command on val dataset
os.system(eval_command)

# Test command for testing on test dataset
test_command = f'python {yolov5_path}/val.py --weights {weights_path}/last.pt --data {data_yaml_path}  --conf 0.5 --iou 0.65'

# Run test command on test dataset
os.system(test_command)
