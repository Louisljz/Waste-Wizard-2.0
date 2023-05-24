import torch
import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image
import numpy as np
import json
import cv2
import time
import os

model = fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 61)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()
transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float)])

with open('annotations.json', 'r') as f:
    annotations = json.loads(f.read())

def lookup_label(catid):
    for item in annotations['categories']:
        if item['id'] == catid:
            return item['supercategory'], item['name']

def draw_bbox(img, pred):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    indices = nms(pred['boxes'], pred['scores'], 0.2)
    for i in indices:
        box = pred['boxes'][i]
        label = pred['labels'][i]
        str_label = lookup_label(label-1)[1]
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped_img = img[y0:y1, x0:x1]
        folder_path = os.path.join('annotations', file_name.split('.')[0])
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, str_label + '.jpg')
        cv2.imwrite(file_path, cropped_img)
    
    print('All annotations saved!')

file_name = '2.jpg'
img = Image.open(os.path.join('images', file_name)).convert('RGB')
t_img, _ = transform(img, target=None)

with torch.no_grad():
    start = time.time()
    pred = model([t_img])
    end = time.time()
    print('Inference Time on CPU: ', round(end-start, 2), 's')

draw_bbox(np.array(img), pred[0])
