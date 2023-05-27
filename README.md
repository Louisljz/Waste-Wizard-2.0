# Waste-Wizard-2.0

Dataset: https://github.com/pedropro/TACO

Approach: Object Detection

Model: Faster-RCNN

Pipeline:

EDA - 20 %

- Get-to-Know annotations.json
- Download Image Data
- Visualize annotations

Preprocessing - 40 %

- Create Dataset Class
- Prepare Transforms
- Split Data into Subsets
- Initiate Data Loaders

Training - 20 %

- Fine-tune Faster-RCNN Head
- Train Model

Deployment - 20 %
- Build Inference Engine

Performance: 

Able to localize various objects in an image, decent classification. 
However, main weakness is the over-sensitivity of predictions, and latency of inference on CPU. 

Solution:

Latency: Use lighter backbone models like MobileNet

Over-sensitivity: Group names to material categories
