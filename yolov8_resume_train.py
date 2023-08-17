import argparse
import os

import comet_ml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='model.pt path(s)')
parser.add_argument('-e', '--epochs', type=int, default=1000, help='number of epochs, default to 10')
parser.add_argument('--exp_key', type=str, default=None, help='experiment key for comet.ml')

args = parser.parse_args()
comet_ml.init(
    api_key=os.getenv("COMET_API_KEY"),
    project_name=os.getenv("COMET_PROJECT_NAME"),
    workspace=os.getenv("COMET_WORKSPACE"),
    # experiment_key=args.exp_key,
)

model = YOLO(args.model)
model.resume = True

model.train(
    patience=10000,
    epochs=50000,
    batch=1,
)
