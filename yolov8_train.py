import argparse
import json
import os
from pathlib import Path

import yaml

from utility import cur_dt_str, args_in_lines

# Try to use comet_ml to log model training, which has to be done before load PyTorch
try:
    import comet_ml
    from comet_ml import Experiment

    # If environment variables are not set, you can set them here
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")
    COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")

    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    # Add date as tag
    experiment.add_tag(f'train_{cur_dt_str().split("-", 1)[0]}')
except ImportError:
    print("comet_ml not installed, model training not logged.")
    experiment = None
except Exception:
    print(
        f"model training not logged since comet failed to initialize.\n"
        f"{COMET_API_KEY=}, {COMET_PROJECT_NAME=}, {COMET_WORKSPACE=}"
    )
    # TODO: give detail why comet failed to initialize. Look for `.comet.config`
    experiment = None
finally:
    import torch
    from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="./gd_s_m5472_best.pt",
    help="Model.pt path(s) or pretrained model name",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="datasets/20240619-155823-gd_s_h/data.yaml",
    help="dataset.yaml path",
)
parser.add_argument(
    "--image_size", type=int, default=5472, help="image size for training"
)
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=5000, help="number of epochs")
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="number of workers for dataloader, default to 8",
)
parser.add_argument(
    "--patience", type=int, default=1000, help="early stopping patience"
)
parser.add_argument("--resume", action="store_true", help="resume training")

args = parser.parse_args()
print(args_in_lines(args))
dataset_file = args.dataset
image_size = args.image_size
batch_size = args.batch_size
epochs = args.epochs
workers = args.workers
base_model = args.model
patience = args.patience
resume = args.resume

# if mps is available, indicate running on mac and no cuda available, use mps
device = "mps" if torch.backends.mps.is_available() else "0"
if batch_size == -1 and device == "mps":
    batch_size = 1  # mps does not support auto-batch size

# modify dataset absolute path in .yaml file according to machine
dataset_cfg = yaml.load(open(dataset_file, "r"), Loader=yaml.FullLoader)
dataset_cfg["path"] = Path(dataset_file).parent.absolute().as_posix()
yaml.dump(dataset_cfg, open(dataset_file, "w"))

# load a model
model = YOLO(base_model)

model_train_param_str = f"m={Path(base_model).stem}_imgsz={image_size}_bs={batch_size}"
model_name = f"d={Path(dataset_file).stem}_{model_train_param_str}_{cur_dt_str()}"
if experiment is not None:
    experiment.set_name(model_name)

# train the model
model.train(
    data=dataset_file,
    optimizer="auto",
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
    project="models",
    name=model_name,
    # resume training
    resume=resume,
)

# run validation once
metrics = model.val(
    # data=, # need to specify dataset.yaml if not in default location
    imgsz=image_size,
    batch=1,
    conf=0.001,  # default from YOLOv8
    iou=0.6,  # default from YOLOv8
    max_det=500,  # default from YOLOv8
    device=device,
    # save to project/name
    project=f"models/{model_name}",
    name="validation",
)  # returns: ultralytics.utils.metrics.Metric
print(f"{metrics.box.maps=}")  # a list contains map50-95 of each category
cls_map50_95 = metrics.box.maps
cls_ap50 = metrics.box.ap50
model_eval_results = {
    "model_name": model_name,
    "datetime": cur_dt_str(),
    "cls_names": ",".join(model.names.values()),
    "model": base_model,
    "dataset": dataset_file,
    "image_size": image_size,
    "batch_size": batch_size,
}
model_eval_results.update(
    {f"{model.names[cls]}_map": map for cls, map in zip(model.names, cls_map50_95)}
)
model_eval_results.update(
    {f"{model.names[cls]}_ap50": ap50 for cls, ap50 in zip(model.names, cls_ap50)}
)
print(f"{model_eval_results=}")

# dump as json
with open(f"models/{model_name}/processed_eval.json", "w") as f:
    json.dump(model_eval_results, f)

print("Done!")
