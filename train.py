import sys
sys.path.append("./")
from ultralytics import YOLO
import optuna
# Load model
model = YOLO("Can choose from ultralytics/cfg/models/anymodel/.yaml")

# Train the model with additional parameters
model.train(
    data="data.yaml",
    epochs=100,                     # Number of training epochs
    batch=16,                 # Batch size for training
    workers=8,
    imgsz=512,                     # Image size (resize images to this size)
    optimizer="EvoFlow",		# Choose based on which optimizer want to use, AdamW, EvoFlow,SGD etc...
    device="0",                    # Use specific GPU (e.g., "0" for first GPU, or "cpu" for CPU)

)
