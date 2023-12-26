import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics import settings

from utility import cur_dt_str, args_in_lines

COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
try:
    import comet_ml
    from comet_ml import Experiment

    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    # add date as tag
    experiment.add_tag(f'train_{cur_dt_str().split("-", 1)[0]}')
except ImportError:
    print('comet_ml not installed, model training not logged.')
    experiment = None
except comet_ml.exceptions.CometException:
    print(f'model training not logged since comet failed to initialize. '
          f'{COMET_API_KEY=}, {COMET_PROJECT_NAME=}, {COMET_WORKSPACE=}')
    experiment = None

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='yolov8m.pt', help='Model.pt path(s) or pretrained model name')
parser.add_argument('-d', '--dataset', type=str,
                    default='/abyss/home/localcode/labelbox-handler/yolov8_labels/mc_dtc2023/mc_dtc2023.yaml',
                    help='dataset.yaml path')
parser.add_argument('--image_size', type=int, default=5472,
                    help='image size, default to original size in blackbuck dataset, 5472')
parser.add_argument('--batch_size', type=int, default=1, help='batch size, default to auto (-1)')
parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs, default to 10')
parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
parser.add_argument('--patience', type=int, default=1000, help='early stopping patience, default to 100')
parser.add_argument('--ontology', type=str, default='all-mc', help='str to indicate ontology, default to empty str')

args = parser.parse_args()
print(args_in_lines(args))
dataset_file = args.dataset
image_size = args.image_size
batch_size = args.batch_size
epochs = args.epochs
workers = args.workers
base_model = args.model
patience = args.patience
ontology = args.ontology

if ontology:
    dataset_name = Path(dataset_file).stem
    dataset_file = dataset_file.replace(dataset_name, f'{dataset_name}_{ontology}')
    print(f"Ontology specified: {dataset_file=}")
    assert Path(dataset_file).exists(), f"{dataset_file} does not exist!"

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
model = YOLO(base_model)

model_train_param_str = f"m={Path(base_model).stem}_imgsz={image_size}_bs={batch_size}"
model_name = f"d={Path(dataset_file).stem}_o={ontology}_{model_train_param_str}_{cur_dt_str()}"
if experiment is not None:
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

# run validation once
metrics = model.val(
    # data=, # need to specify dataset.yaml if not in default location
    imgsz=image_size,
    batch=1,
    conf=0.001,  # default
    iou=0.6,  # default
    max_det=500,  # default
    device=device,
    # save to project/name
    project=f'models/{model_name}',
    name='validation',
)  # returns: ultralytics.utils.metrics.Metric
print(f"{metrics.box.maps=}")  # a list contains map50-95 of each category
cls_map50_95 = metrics.box.maps
cls_ap50 = metrics.box.ap50
model_eval_results = {
    'model_name': model_name,
    'datetime': cur_dt_str(),
    'ontology': ontology,
    'cls_names': ','.join(model.names.values()),
    'model': base_model,
    'dataset': dataset_file,
    'image_size': image_size,
    'batch_size': batch_size,
}
model_eval_results.update({
    f"{model.names[cls]}_map": map for cls, map in zip(model.names, cls_map50_95)
})
model_eval_results.update({
    f"{model.names[cls]}_ap50": ap50 for cls, ap50 in zip(model.names, cls_ap50)
})
print(f"{model_eval_results=}")

# dump as json
with open(f'models/{model_name}/processed_eval.json', 'w') as f:
    json.dump(model_eval_results, f)

print("Done!")
