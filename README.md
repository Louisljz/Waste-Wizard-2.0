# Waste-Wizard-2.0

## General Info
- Dataset: https://github.com/pedropro/TACO
- Colab Notebook: https://drive.google.com/drive/folders/1K6u6DM8RirkAsTYxmbG-t1f6an99TjuL?usp=sharing
- Streamlit Webapp: https://louisljz-wastewizard2.streamlit.app/
- Approach: Object Detection
- Model: Faster-RCNN

## Pipeline:
### EDA - 20 %
- Get-to-Know annotations.json
- Download Image Data
- Visualize annotations

### Preprocessing - 40 %
- Create Dataset Class
- Prepare Transforms
- Split Data into Subsets
- Initiate Data Loaders

### Training - 20 %
- Fine-tune Faster-RCNN Head
- Train Model

### Deployment - 20 %
- Build Inference Engine

## Performance: 
Able to localize various objects in an image, decent classification. 
However, main weakness is the over-sensitivity of predictions, and latency of inference on CPU. 

Over-sensitivity Solution: Group names to material categories
