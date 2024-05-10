import torch
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import os

image_path = 'C:/Users/RO100202/OneDrive - ANRITSU CORPORATION/Desktop/example.jpg'

#command = f'python C:/Users/RO100202/yolov5/detect.py --source "{image_path}" --weights C:/Users/RO100202/yolov5/runs/train/my_model3/weights/last.pt --conf 0.25'
command = f'python C:/Users/RO100202/yolov5/detect.py --source "{image_path}" --weights "C:/Users/RO100202/yolov5/runs/train/my_model/weights/last.pt" --conf 0.5 --data "C:/Users/RO100202/pythonProject/models/yolov5s.yaml" --save-txt --save-conf'

os.system(command)

"""# Display the image with bounding boxes
output_image_path = 'C:/Users/RO100202/yolov5/runs/detect/exp/example.jpg'  # Path to the output image with bounding boxes
output_image = Image.open(output_image_path)
plt.figure(figsize=(10, 8))
plt.imshow(output_image)
plt.axis('off')
plt.show()"""


