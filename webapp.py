import streamlit as st
from PIL import Image
import torch
import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import numpy as np
import json
import cv2
import time
import os

st.set_page_config('Waste Wizard 2.0', ':recycle:')
st.title('Waste Wizard 2.0 :recycle:')
folder_path = os.path.dirname(__file__)

model = fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 61)
model_path = os.path.join(folder_path, 'model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float)])

anot_path = os.path.join(folder_path, 'annotations.json')
with open(anot_path, 'r') as f:
    annotations = json.loads(f.read())

cat_path = os.path.join(folder_path, 'categories.json')
with open(cat_path, 'r') as f:
    categories = json.loads(f.read())

def lookup_label(catid):
    cat_name = 'others'
    for item in annotations['categories']:
        if item['id'] == catid:
            name = item['name']
            break
    
    for cat in categories:
        for item in categories[cat]:
            if item == name:
                cat_name = cat
                break
    
    return cat_name, name

def draw_bbox(img, pred):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    indices = nms(pred['boxes'], pred['scores'], 0.2)
    for i in indices:
        box = pred['boxes'][i]
        label = pred['labels'][i]
        str_labels = lookup_label(label-1)
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped_img = img[y0:y1, x0:x1]
        st.image(cropped_img, str_labels[0] + ': ' + str_labels[1])

buffer = st.camera_input('Take a photo of trash objects')
if buffer:
    img = Image.open(buffer).convert('RGB')
    t_img, _ = transform(img, target=None)

    with torch.no_grad():
        start = time.time()
        pred = model([t_img])
        end = time.time()

    if pred[0]['boxes'].tolist():
        draw_bbox(np.array(img), pred[0])
        st.info(f'Inference Time on CPU: {round(end-start, 2)} s')
    else:
        st.error('No objects found!')
