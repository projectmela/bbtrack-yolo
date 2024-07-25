import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from utility import cur_dt_str

model_paths = [
    # /scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd
    # /scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc
    # models 20231117 mc
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231118-005142/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231118-005145/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231118-012653/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=2560_bs=4_20231118-005146/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231118-014426/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231118-014225/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=5472_bs=1_20231118-005137/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231118-005143/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=5472_bs=1_20231118-005143/weights/best.pt",
    # models 20231122 gd
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=2560_bs=4_20231122-050555/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231122-040301/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231122-054018/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231122-073256/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231122-061030/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231122-054203/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=5472_bs=1_20231122-035129/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231122-035009/weights/best.pt",
    # "/scratch/cs/css-llm/jonathan/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=5472_bs=1_20231122-024757/weights/best.pt",

    # models_20231219_ontology
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_all-mc_o=all-mc_m=yolov8m_imgsz=5472_bs=1_20231219-055807/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_all-mc_o=all-mc_m=yolov8x_imgsz=5472_bs=1_20231219-055807/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd_o=gd_m=yolov8m_imgsz=5472_bs=1_20231219-061150/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd_o=gd_m=yolov8x_imgsz=5472_bs=1_20231219-055807/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-drone-bird_o=gd-drone-bird_m=yolov8m_imgsz=5472_bs=1_20231219-054010/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-drone-bird_o=gd-drone-bird_m=yolov8x_imgsz=5472_bs=1_20231219-054003/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_20231219-054003/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8x_imgsz=5472_bs=1_20231219-054002/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_rm-unknown_o=rm-unknown_m=yolov8m_imgsz=5472_bs=1_20231219-054002/weights/best.pt",
    # "/abyss/home/localcode/bb-yolo/models/models_20231219_ontology/d=mc_dtc2023_rm-unknown_o=rm-unknown_m=yolov8x_imgsz=5472_bs=1_20231219-054002/weights/best.pt",

    # sc, mc, gd
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231118-014225/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=2560_bs=4_20231118-005146/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231118-005143/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231118-014426/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231118-005145/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8n_imgsz=5472_bs=1_20231118-005143/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231118-012653/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231118-005142/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231117_mc/d=mc_dtc2023_m=yolov8x_imgsz=5472_bs=1_20231118-005137/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231209-013850/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8m_imgsz=2560_bs=4_20231209-013749/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231209-013749/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231209-022311/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231209-013849/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8n_imgsz=5472_bs=1_20231209-013748/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231209-013850/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231209-013749/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_20231209_sc/d=sc_dtc2023_m=yolov8x_imgsz=5472_bs=1_20231209-013749/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=1280_bs=16_20231122-061030/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=2560_bs=4_20231122-050555/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8m_imgsz=5472_bs=1_20231122-035009/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231122-073256/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=2560_bs=4_20231122-054018/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8n_imgsz=5472_bs=1_20231122-035129/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=1280_bs=16_20231122-054203/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=2560_bs=4_20231122-040301/weights/best.pt",
    "/abyss/home/localcode/bb-yolo/models/models_202311122_gd/d=gd_dtc2023_m=yolov8x_imgsz=5472_bs=1_20231122-024757/weights/best.pt",
]

dataset_dir = Path("/abyss/home/localcode/labelbox-handler/yolov8_labels")
# model_paths += [model_path.replace("best.pt", "last.pt") for model_path in model_paths]
print(f"{model_paths=}")
eval_results = []
eval_dt = cur_dt_str()
for model_path in model_paths:
    model = YOLO(model_path)
    print(f"{model.names=}")

    # set batch size
    if "imgsz=5472" in model_path:
        batch_size = 1
    elif "imgsz=2560" in model_path:
        batch_size = 4
    elif "imgsz=1280" in model_path:
        batch_size = 16
    else:
        raise ValueError(f"Unknown imgsz in {model_path=}")
    dataset_name = model_path.split("d=")[1].split("_")[0]
    metrics = model.val(
        # need to specify dataset.yaml if not in default location
        data=(dataset_dir / dataset_name / f"{dataset_name}.yaml"),
        batch=1,
    )  # returns: ultralytics.utils.metrics.Metric
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
    eval_results.append(model_eval_results)

    eval_results_df = pd.DataFrame(eval_results)
    eval_results_df.to_parquet(f"eval_results_{eval_dt}.parquet")
    eval_results_df.to_csv(f"eval_results_{eval_dt}.csv")

    # dump as json
    with open(f"eval_results_{eval_dt}.json", "w") as f:
        json.dump(eval_results, f)
