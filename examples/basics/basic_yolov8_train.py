import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from bbtrack_yolo.util import cur_dt_str, args_in_lines

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="yolov8n.pt",
    help="Model.pt path(s) or pretrained model name",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="datasets/gd_toy/data.yaml",
    help="dataset.yaml path",
)

args = parser.parse_args()
print(args_in_lines(args))
dataset_file = args.dataset
base_model = args.model
image_size = 640
batch_size = 1

# modify dataset absolute path in .yaml file according to machine
dataset_cfg = yaml.load(open(dataset_file, "r"), Loader=yaml.FullLoader)
dataset_cfg["path"] = Path(dataset_file).parent.absolute().as_posix()
yaml.dump(dataset_cfg, open(dataset_file, "w"))

# load a model
model = YOLO(base_model)

model_train_param_str = f"m={Path(base_model).stem}_imgsz={image_size}_bs={batch_size}"
model_name = f"d={Path(dataset_file).stem}_{model_train_param_str}_{cur_dt_str()}"

# train the model
model.train(
    data=dataset_file,
    optimizer="auto",
    device="mps" if torch.backends.mps.is_available() else "0",
    epochs=10,
    patience=5,  # early stopping patience
    batch=batch_size,
    seed=0,
    deterministic=True,  # reproducible
    imgsz=image_size,
    val=True,  # validate during training
    # save to project/name
    project="models",
    name=model_name,
)
