import argparse
import os
from pathlib import Path

import comet_ml
import torch
import yaml
from comet_ml import Experiment
from ultralytics import YOLO
from ultralytics import settings

from utility import cur_dt_str

experiment = Experiment(
    api_key=os.getenv("COMET_API_KEY"),
    project_name=os.getenv("COMET_PROJECT_NAME"),
    workspace=os.getenv("COMET_WORKSPACE"),
)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='yolov8l.pt', help='Model.pt path(s) or pretrained model name')
parser.add_argument('-d', '--dataset', type=str, default='dataset/bb_2022/bb_2022.yaml', help='dataset.yaml path')
parser.add_argument('--image_size', type=int, default=5472,
                    help='image size, default to original size in blackbuck dataset, 5472')
parser.add_argument('--batch_size', type=int, default=-1, help='batch size, default to auto (-1)')
parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs, default to 10')
parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience, default to 100')

args = parser.parse_args()
dataset_file = args.dataset
image_size = args.image_size
batch_size = args.batch_size
epochs = args.epochs
workers = args.workers
model = args.model
patience = args.patience

# TODO: check if this is necessary
settings.update({
    'runs_dir': './runs',
    'weights_dir': './models',
})

# if mps is available, indicate running on mac and no cuda available, use mps
device = 'mps' if torch.backends.mps.is_available() else '0'
if batch_size == -1 and device == 'mps':
    batch_size = 1  # mps does not support auto-batch size

# modify dataset absolute path in .yaml file according to machine
dataset_cfg = yaml.load(open(dataset_file, 'r'), Loader=yaml.FullLoader)
dataset_cfg['path'] = Path(dataset_file).parent.absolute().as_posix()
yaml.dump(dataset_cfg, open(dataset_file, 'w'))

# load a model
model = YOLO(model)

model_train_param_str = f"m={Path(args.model).stem}_imgsz={image_size}_bs={batch_size}"
model_name = f"d={Path(dataset_file).stem}_{model_train_param_str}_{cur_dt_str()}"
experiment.set_name(model_name)

# train the model
model.train(
    data=dataset_file,
    optimizer='auto',
    device=device,
    epochs=epochs,
    patience=patience,  # early stopping patience
    batch=batch_size,
    workers=8,  # number of workers threads for dataloader
    save_period=int(0.1 * epochs),  # save model snapshots every 10% of epochs
    seed=0,
    deterministic=True,  # reproducible
    imgsz=image_size,
    val=True,  # validate during training
    # save to project/name
    project='models',
    name=model_name,
)

model.val(
    imgsz=image_size,
    batch=batch_size if batch_size != -1 else 1,  # validation can not be auto-batch size
    conf=0.001,  # default
    iou=0.6,  # default
    max_det=300,  # default
    device=device,
    # save to project/name
    project=f'models/{model_name}',
    name='validation',
)  # run validation once
