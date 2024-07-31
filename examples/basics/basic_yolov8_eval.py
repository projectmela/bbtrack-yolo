""" Evaluate a model on a dataset and print out evaluation results. """

from ultralytics import YOLO

from bbtrack_yolo.util import cur_dt_str

# Manually set arguments
model_path = ""  # ../../xxxx.pt
dataset_yaml_path = ""  # ../../xxxx.yaml, the dataset that the model was trained on
eval_dt = cur_dt_str()  # current datetime string

# Load model
model = YOLO(model_path)
print(f"{model.names=}")

# Evaluate model
metrics = model.val(
    # need to specify dataset.yaml if not in default location
    data=dataset_yaml_path,
    batch=1,
)  # returns: ultralytics.utils.metrics.Metric

# Wrap evaluation results
print(f"{metrics.box.maps=}")  # a list contains map50-95 of each category
cls_map50_95 = metrics.box.maps
cls_ap50 = metrics.box.ap50
model_eval_results = {"model_path": model_path, "datetime": cur_dt_str()}
model_eval_results |= {
    f"{model.names[cls]}_map": map for cls, map in zip(model.names, cls_map50_95)
}
model_eval_results |= {
    f"{model.names[cls]}_ap50": ap50 for cls, ap50 in zip(model.names, cls_ap50)
}
print(f"{model_eval_results=}")
