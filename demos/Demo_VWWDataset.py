#VWW dataset path: visualwakewords/script/annotation_vww
import torch
import pyvww
from pyvww.utils import VisualWakeWords
import numpy as np

from gluoncv import data, utils
from matplotlib import pyplot as plt





train_dataset = data.COCODetection('.',splits=['/Volumes/DS/Priya_Desktop/annotations_2/instances_train2017'])
print('Num of training images:', len(train_dataset))
val_dataset = data.COCODetection('.',splits=['/Volumes/DS/Priya_Desktop/annotations_2/instances_val2017'])





print(len(train_dataset))
train_image, train_label = train_dataset[37572]

bounding_boxes = train_label[:, :4]
class_ids = train_label[:, 4:5]
print(train_label.shape)
print('Image size (height, width, RGB):', train_image.shape)
print('Num of objects:', bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n',
      bounding_boxes)
print('Class IDs (num_boxes, ):\n', class_ids)


utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=train_dataset.classes)
plt.show()