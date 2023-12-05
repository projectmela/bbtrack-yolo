""" run YOLOv8 inferences, save results to csv, parquet and MOT17 submission format"""
import argparse
import traceback
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from ultralytics import YOLO

from utility import cur_dt_str

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='model.pt path')
parser.add_argument('-s', '--source', type=str, required=True, help='source to predict')
parser.add_argument('--batch_size', type=int, default=1, help='batch size, default to 1')
parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
parser.add_argument('--save_dir', type=str, default='predictions', help='save directory, default to "predictions"')
parser.add_argument('--plot', action='store_true', help='save plotted results to save_dir')
parser.add_argument('--pred_th', type=float, default=0.0, help='prediction threshold, default to 0.0')

args = parser.parse_args()
model_file = args.model
source = args.source
batch_size = args.batch_size
workers = args.workers
save_dir = args.save_dir
plot = args.plot
pred_th = args.pred_th

assert Path(model_file).exists(), f'Model {model_file} does not exist'
model = YOLO(model_file)

# generate string describing model and dataset to name saving directory
model_name = Path(model_file).parent.parent.name
source_name = Path(source).stem
dest_name = f'{model_name}_{source_name}_{cur_dt_str()}'

device = 'mps' if torch.backends.mps.is_available() else '0'  # choose mps if available
# inference on source
results = model.predict(
    source,
    # path parameters
    project=save_dir,
    name=dest_name,
    # inference parameters
    stream=True,  # avoid memory overflow
    device=device,
    conf=pred_th,
    max_det=500,
    # visualization parameters
    save=plot,  # save plotted results to save_dir
    line_width=3,
    show_labels=True,
    save_frames=False,
    save_conf=True,
    save_txt=False,  # saves preds to .txt, one file per frame
)  # in stream mode, return a generator

# open csv file
dest = Path(args.save_dir) / dest_name
dest.mkdir(parents=True, exist_ok=True)
csv_path = dest / 'yolov8_dets.csv'
csv_file = open(csv_path, 'w')
# write csv header
csv_file.write("file_path,frame,id,"
               "bb_left,bb_top,bb_width,bb_height,"
               "conf,cls,cls_name\n")

# collect results into human readable dataframes
frame_id = 1  # frame id starts from 1
last_path = None
try:
    for result in results:
        result = result.cpu()  # move result to cpu

        # set frame_id according to file_path
        if result.path == last_path:  # different image from same video, increment id
            frame_id += 1
        else:  # a new video or image, reset id
            frame_id = 1
            last_path = result.path

        # save box detections to csv
        for box in result.numpy().boxes:
            bbox = box.xywh.reshape(-1)
            bbl, bbt = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2
            bbw, bbh = bbox[2], bbox[3]
            conf, cls = box.conf.item(), box.cls.item()
            cls_name = model.names[cls] if model.names is not None else str(cls)
            csv_file.write(f"\"{result.path}\","
                           f"{frame_id},"
                           f"-1,"  # dummy id
                           f"{bbl},{bbt},{bbw},{bbh},"
                           f"{conf},{cls},{cls_name}\n")
except Exception as e:  # catch all exceptions, stop inference but save results
    logger.error(f'Error occurred during inference:\n{traceback.format_exc()}')
    logger.info(f'Still try to convert results ...')
finally:
    csv_file.close()
    # post process results
    logger.info(f'Converting results to csv ...')
    result_df = pd.read_csv(csv_path)
    try:
        result_df.to_parquet(dest / 'yolov8_dets.parquet')
    except ImportError as e:
        print(f"'pyarrow' or 'fastparquet' not found, only saving to csv instead. See error: {e}")

    logger.info("Saving blackbuck dets results to mot format ...")

    # convert to mot17 format:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    result_df.loc[:, ['x', 'y', 'z']] = -1  # add dummy values for x, y, z
    # save all classes
    (result_df
     .loc[:, ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']]
     .to_csv(dest / 'yolov8_dets_mot.txt', index=False, header=False))
    # save only blackbuck classes
    bb_cls_name = ['bb', 'bbfemale', 'bbmale']
    bb_df = result_df[result_df["cls_name"].isin(bb_cls_name)]
    if bb_df.empty:
        logger.warning(f"No blackbuck ({bb_cls_name}) detected in {source}.\n"
                       f"Classes detected: {result_df['cls_name'].unique()}")
    else:
        (bb_df
         .loc[:, ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']]
         .to_csv(dest / 'yolov8_dets_bb_mot.txt', index=False, header=False))
