import json

from ultralytics import YOLO
from utility import cur_dt_str

model_paths = [
    # /scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd
    # /scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc
    # models 20231117 mc
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231118-005142/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231118-005145/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231118-012653/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=2560_bs=4_20231118-005146/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231118-014426/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231118-014225/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=5472_bs=1_20231118-005137/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231118-005143/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=5472_bs=1_20231118-005143/weights/best.pt",

    # models 20231122 gd
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=2560_bs=4_20231122-050555/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231122-040301/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231122-054018/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231122-073256/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231122-061030/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231122-054203/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=5472_bs=1_20231122-035129/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231122-035009/weights/best.pt",
    "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=5472_bs=1_20231122-024757/weights/best.pt",
]

eval_results = []
for model_path in model_paths:
    model = YOLO(model_path)
    print(f"{model.names=}")
    # ultralytics.utils.metrics.Metric
    metrics = model.val()  # need to specify data=path_to.yaml
    print(f"{metrics.box.maps=}")  # a list contains map50-95 of each category
    cls_map = metrics.box.maps
    model_eval_results = {'model_path': model_path}
    model_eval_results |= [
        [model.names[cls], map] for cls, map in zip(model.names, cls_map)
    ]
    print(f"{model_eval_results=}")
    eval_results.append(model_eval_results)

    # dump as json
    with open(f'eval_results_{cur_dt_str()}.json', 'w') as f:
        json.dump(eval_results, f)
