import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='model.pt path(s)')

args = parser.parse_args()

model = YOLO(args.model)

# Resume training
model.train(resume=True)
