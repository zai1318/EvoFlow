
# EvoFlow: A Robust Optimizer for Deep Learning via Gradient Consistency and Evolutionary Strategies

Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

## Description
This repository implements EvoFlow, a hybrid optimizer that enhances convergence stability and generalization in deep neural networks by combining gradient consistency adaptation with evolutionary perturbations. EvoFlow dynamically adjusts momentum and smoothing parameters based on gradient alignment and periodically perturbs model parameters to escape sharp local minima. It is evaluated on object detection (YOLOv5, YOLOv9, YOLOv11 on COCO) and image classification (ResNet-18 on CIFAR-10), outperforming baselines like SGD, AdamW, RAdam, and RMSProp.

## Getting Started

### Prerequisites
- Python 3.8
- PyTorch 1.12
- NVIDIA GPU (e.g., RTX 3090) with CUDA support
- Datasets: 
  - COCO 2017 (object detection)
  - CIFAR-10 (image classification)

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/zai1318/EvoFlow.git
cd EvoFlow
pip install -r requirements.txt
```

Download datasets:
- **COCO 2017**: Available at [http://cocodataset.org/](http://cocodataset.org/). Extract to `data/coco/` with subdirectories `train2017`, `val2017`, and `annotations`.
- **CIFAR-10**: Already available in resnet_EvoFlow Folder.

## Usage
### Training
- **Object Detection (YOLO on COCO)**:
  ```bash
  python train.py
  ```
  - Located in the root directory.
  - Trains a YOLO model (e.g., YOLOv9) for 100 epochs with a batch size of 16 using EvoFlow. Preconfigured in `train.py` with:
    - `data="data.yaml"` (edit to match your COCO paths)
    - `epochs=100`
    - `batch=16`
    - `imgsz=512`
    - `optimizer="EvoFlow"`
    - `device="0"` (first GPU)
  - To use a different model, edit `train.py` to load from `ultralytics/cfg/models/` (e.g., `model = YOLO("ultralytics/cfg/models/v9/yolov9.yaml")`).
  - To use another optimizer (e.g., AdamW, SGD), change `optimizer="EvoFlow"` to any Ultralytics-supported optimizer (e.g., `optimizer="AdamW"`).

- **Image Classification (ResNet-18 on CIFAR-10)**:
  ```bash
  cd EvoFlow/resnet_EvoFlow
  python train.py
  ```
  - Trains ResNet-18 with five optimizers (EvoFlow, AdamW, SGD, RAdam, RMSProp) sequentially, each for 100 epochs with a batch size of 128, using standard augmentation (random cropping, flipping, normalization).
  - Uses CIFAR-10 files in `EvoFlow/resnet_EvoFlow`.
  - Outputs:
    - Console: Training loss per epoch, final top-1/top-5 accuracy, inference time, and a comparison table.
    - Files: `optimizer_comparison.yaml` (results), `training_loss_comparison.png` (loss plot).
  
EvoFlow hyperparameters (hardcoded in both scripts):
- Learning rate: 0.001
- Momentum decay ($\beta_1$): 0.9
- Second-moment decay ($\beta_2$): 0.999
- Weight decay: 0.0001
- Smoothing rate ($\alpha$): 0.85
- Perturbation frequency ($K$): 25
- Perturbation scale ($\sigma$): 0.01 (YOLO) or 0.008 (ResNet)
- Stability constant ($\epsilon$): 1e-7


## Results

### Object Detection on COCO
| Model   | Optimizer | Precision (%) | Recall (%) | mAP@0.5 | mAP@0.5:0.95 |
|---------|-----------|---------------|------------|---------|--------------|
| YOLOv5  | AdamW     | 43.8          | 10.2       | 9.7     | 5.4          |
| YOLOv5  | EvoFlow   | 35.1          | 21.8       | 22.0    | 14.4         |
| YOLOv9  | AdamW     | 33.7          | 25.6       | 25.8    | 17.0         |
| YOLOv9  | EvoFlow   | 52.3          | 29.6       | 33.3    | 22.8         |
| YOLOv11 | AdamW     | 48.4          | 10.7       | 7.9     | 4.5          |
| YOLOv11 | EvoFlow   | 39.0          | 24.2       | 24.6    | 15.5         |

### Image Classification on CIFAR-10
| Optimizer | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Inference Time (ms/image) |
|-----------|--------------------|--------------------|---------------------------|
| SGD       | 90.76              | 99.58              | 1.8                       |
| AdamW     | 92.34              | 99.77              | 2.3                       |
| RAdam     | 75.41              | 98.52              | 1.9                       |
| RMSProp   | 91.26              | 99.65              | 2.5                       |
| EvoFlow   | 92.28              | 99.77              | 2.3                       |

Results match Tables 3 and 4 in the NeurIPS submission. Inference times measured on an NVIDIA RTX 3090 GPU.

## Contributing
This is a submission for NeurIPS 2025 review. Post-review, contributions or issues can be submitted via GitHub Issues to improve the implementation.

## License
This project is licensed under the MIT License (see `LICENSE` file).

## Citation
Please cite this work as:
```
@article{evoflow_neurips2025,
  title={EvoFlow: A Robust Optimizer for Deep Learning via Gradient Consistency and Evolutionary Strategies},
  author={Anonymous},
  journal={Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025}
}
HEAD
Initial commit of EvoFlow-Optimizer
