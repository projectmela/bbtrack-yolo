import json
import os

from ultralytics import YOLO

from utility import cur_dt_str

model_paths = [
    # /scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd
    # /scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc
    # models_20231219_ontology
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd_o=gd_m=yolov8m_imgsz=5472_bs=1_20231219-061150",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd_o=gd_m=yolov8x_imgsz=5472_bs=1_20231219-055807",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_all-mc_o=all-mc_m=yolov8x_imgsz=5472_bs=1_20231219-055807",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_all-mc_o=all-mc_m=yolov8m_imgsz=5472_bs=1_20231219-055807",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8x_imgsz=5472_bs=1_20231219-054002",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-drone-bird_o=gd-drone-bird_m=yolov8x_imgsz=5472_bs=1_20231219-054003",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_rm-unknown_o=rm-unknown_m=yolov8x_imgsz=5472_bs=1_20231219-054002",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-drone-bird_o=gd-drone-bird_m=yolov8m_imgsz=5472_bs=1_20231219-054010",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_20231219-054003",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_rm-unknown_o=rm-unknown_m=yolov8m_imgsz=5472_bs=1_20231219-054002",
    # models_20231209_sc
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231209-013849",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231209-013749",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231209-022311",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231209-013850",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231209-013850",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231209-013749",
]

model_paths = [os.path.join(p, 'weights', 'best.pt') for p in model_paths]
model_paths += [model_path.replace('best.pt', 'last.pt') for model_path in model_paths]
newline = '\n'
print(f"model_paths:\n{newline.join(model_paths)}")
eval_results = []
eval_dt = cur_dt_str()
for model_path in model_paths:
    model = YOLO(model_path)
    print(f'load model from {model_path}')
    print(f"{model.names=}")

    # set batch size
    if 'imgsz=5472' in model_path:
        batch_size = 1
    elif 'imgsz=2560' in model_path:
        batch_size = 4
    elif 'imgsz=1280' in model_path:
        batch_size = 16
    else:
        raise ValueError(f"Unknown imgsz in {model_path=}")

    metrics = model.val(
        # data=, # need to specify dataset.yaml if not in default location
        batch=1,
    )  # returns: ultralytics.utils.metrics.Metric
    print(f"{metrics.box.maps=}")  # a list contains map50-95 of each category
    cls_map50_95 = metrics.box.maps
    cls_ap50 = metrics.box.ap50
    model_eval_results = {'model_path': model_path, 'datetime': cur_dt_str()}
    model_eval_results.update({
        f"{model.names[cls]}_map": map for cls, map in zip(model.names, cls_map50_95)
    })
    model_eval_results.update({
        f"{model.names[cls]}_ap50": ap50 for cls, ap50 in zip(model.names, cls_ap50)
    })
    print(f"{model_eval_results=}")
    eval_results.append(model_eval_results)

    # dump as json
    with open(f'eval_results_{eval_dt}.json', 'w') as f:
        json.dump(eval_results, f)
