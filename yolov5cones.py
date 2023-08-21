#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


get_ipython().system('ln -s /content/gdrive/My\\ Drive/ /mydrive')


# In[ ]:


get_ipython().run_line_magic('cd', '/mydrive/yolov5trafficcones')


# In[ ]:


# clone YOLOv5 repository
get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone repo')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().system('git reset --hard fbe67e465375231474a2ad80a4389efc77ecff99')


# In[ ]:


# install dependencies as necessary
get_ipython().system('pip install -qr requirements.txt  # install dependencies (ignore errors)')
import torch

from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# In[ ]:


get_ipython().system('pip install -q roboflow')
from roboflow import Roboflow
rf = Roboflow(api_key="TWGQQK1JqEHCyWZ690ob", model_format="yolov5", notebook="roboflow-yolov5")


# In[ ]:


get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().system('pip install roboflow')

from roboflow import Roboflow
rf = Roboflow(api_key="TWGQQK1JqEHCyWZ690ob")
project = rf.workspace("robotica-xftin").project("traffic-cones-4laxg")
dataset = project.version(2).download("yolov5")


# In[ ]:


# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
get_ipython().run_line_magic('cat', '{dataset.location}/data.yaml')


# In[ ]:


# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])


# In[ ]:


#this is the model configuration we will use for our tutorial
get_ipython().run_line_magic('cat', '/content/gdrive/MyDrive/yolov5trafficcones/yolov5/models/yolov5s.yaml')


# In[ ]:


#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


# In[ ]:


get_ipython().run_cell_magic('writetemplate', '/content/gdrive/MyDrive/yolov5trafficcones/yolov5/models/yolov5s.yaml', "\n# parameters\nnc: {num_classes}  # number of classes\ndepth_multiple: 0.33  # model depth multiple\nwidth_multiple: 0.50  # layer channel multiple\n\n# anchors\nanchors:\n  - [10,13, 16,30, 33,23]  # P3/8\n  - [30,61, 62,45, 59,119]  # P4/16\n  - [116,90, 156,198, 373,326]  # P5/32\n\n# YOLOv5 backbone\nbackbone:\n  # [from, number, module, args]\n  [[-1, 1, Focus, [64, 3]],  # 0-P1/2\n   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4\n   [-1, 3, BottleneckCSP, [128]],\n   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8\n   [-1, 9, BottleneckCSP, [256]],\n   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16\n   [-1, 9, BottleneckCSP, [512]],\n   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32\n   [-1, 1, SPP, [1024, [5, 9, 13]]],\n   [-1, 3, BottleneckCSP, [1024, False]],  # 9\n  ]\n\n# YOLOv5 head\nhead:\n  [[-1, 1, Conv, [512, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 6], 1, Concat, [1]],  # cat backbone P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 13\n\n   [-1, 1, Conv, [256, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 4], 1, Concat, [1]],  # cat backbone P3\n   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)\n\n   [-1, 1, Conv, [256, 3, 2]],\n   [[-1, 14], 1, Concat, [1]],  # cat head P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)\n\n   [-1, 1, Conv, [512, 3, 2]],\n   [[-1, 10], 1, Concat, [1]],  # cat head P5\n   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)\n\n   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)\n  ]\n")


# In[ ]:


# train yolov5s on custom data for 100 epochs
# time its performance
get_ipython().run_line_magic('%time', '')
get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/yolov5trafficcones/yolov5/')
get_ipython().system("python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/yolov5s.yaml --weights '' --name yolov5s_results  --cache")


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs')


# In[ ]:


from utils.plots import plot_results  # plot results.txt as results.png
Image(filename='/content/gdrive/MyDrive/yolov5trafficcones/yolov5/runs/train/yolov5s_results/results.png', width=1000)  # view results.png


# In[ ]:


cd /content/gdrive/MyDrive/yolov5trafficcones/yolov5


# In[ ]:


print("GROUND TRUTH TRAINING DATA:")
Image(filename='/content/gdrive/MyDrive/yolov5trafficcones/yolov5/runs/train/yolov5s_results/val_batch0_pred.jpg', width=900)


# In[ ]:


print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename='/content/gdrive/MyDrive/yolov5trafficcones/yolov5/runs/train/yolov5s_results/train_batch0.jpg', width=900)


# In[ ]:


get_ipython().run_line_magic('cd', '/content/gdrive/MyDrive/yolov5trafficcones/yolov5/')
get_ipython().system('python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source ../yolov5/Traffic-Cones-2/test/images')


# In[ ]:


import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/gdrive/MyDrive/yolov5trafficcones/yolov5/runs/detect/exp2/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")


# In[ ]:


get_ipython().system('python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source /content/gdrive/MyDrive/yolov5trafficcones/yolov5/Traffic-Cones-2/testvids/test1.mp4')


# In[ ]:


get_ipython().system('python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source /content/gdrive/MyDrive/yolov5trafficcones/yolov5/Traffic-Cones-2/testvids/test2.mp4')


# In[ ]:


get_ipython().system('python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source /content/gdrive/MyDrive/yolov5trafficcones/yolov5/Traffic-Cones-2/testvids/test3.mp4')


# In[ ]:


get_ipython().system('python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source /content/gdrive/MyDrive/yolov5trafficcones/yolov5/Traffic-Cones-2/testvids/test4.mp4')


# In[ ]:




