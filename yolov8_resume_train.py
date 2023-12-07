import argparse
import os
from pathlib import Path

import comet_ml
from comet_ml import Experiment
from ultralytics import YOLO

from utility import cur_dt_str

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='model.pt path')
parser.add_argument('-d', '--dataset', type=str, default='dataset/bb_2022/bb_2022.yaml', help='dataset.yaml path')
parser.add_argument('-e', '--epochs', type=int, default=1000, help='number of epochs, default to 1000')
parser.add_argument('--batch_size', type=int, default=-1, help='batch size, default to auto (-1)')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience, default to 100')

args = parser.parse_args()
model = args.model
dataset = args.dataset
batch_size = args.batch_size
epochs = args.epochs
patience = args.patience

model_name = f"resume_{Path(model).parent.parent.stem}_{cur_dt_str()}"
print(f"model_name: {model_name}")

experiment = Experiment(
    api_key=os.getenv("COMET_API_KEY"),
    project_name=os.getenv("COMET_PROJECT_NAME"),
    workspace=os.getenv("COMET_WORKSPACE"),
)
experiment.set_name(model_name)

model = YOLO(model)

model.train(
    # data=dataset,
    patience=patience,
    epochs=epochs,
    batch=batch_size,
    resume=True,
)