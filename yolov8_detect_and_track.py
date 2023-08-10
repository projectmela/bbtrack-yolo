""" Script to run inference on images, videos, or folder/url of images & videos using YOLOv8 """
import argparse
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from utility import cur_dt_str

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='models/bb_2022/bb_2022.pt', help='model.pt path(s)')
parser.add_argument('-s', '--source', type=str, default='dataset/bb_2022/bb_2022.yaml', help='source to predict')
parser.add_argument('--batch_size', type=int, default=1, help='batch size, default to 1')
parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
parser.add_argument('--save_dir', type=str, default='predictions', help='save directory, default to "predictions"')
parser.add_argument('--plot', action='store_true', help='save plotted results to save_dir')

args = parser.parse_args()
model_file = args.model
source = args.source
batch_size = args.batch_size
workers = args.workers
save_dir = args.save_dir
plot = args.plot

assert Path(model_file).exists(), f'Model {model_file} does not exist'
model = YOLO(model_file)

# generate string describing model and dataset to name saving directory
model_name = Path(model_file).parent.parent.name
source_name = Path(source).stem
dest_name = f'{model_name}_{source_name}_{cur_dt_str()}'
dest = Path(args.save_dir) / dest_name
dest.mkdir(parents=True, exist_ok=True)

device = 'mps' if torch.backends.mps.is_available() else '0'  # choose mps if available
# inference on source
results = model.track(
    source,
    stream=True,  # avoid memory overflow
    device=device,
    save=plot, # save plotted results to save_dir
    show_labels=True,
    line_width=3,
    project=save_dir,
    name=dest_name,
    # tracker="bytetrack.yaml",
    tracker="botsort.yaml",
    conf=0.3,
    iou=0.5,
)  # in stream mode, return a generator
results = list(results)  # convert to list, trigger to run inference

# collect results into human readable dataframes
result_dfs = []
frame_id = 0
last_path = None
for result in results:

    file_path = result.path

    if file_path != last_path:  # a new video or image, reset id
        frame_id = 0
        last_path = file_path
    else:  # different image from same video, increment id
        frame_id += 1

    boxes = result.boxes.xywh.to('cpu').numpy()
    conf = result.boxes.conf.to('cpu').numpy()
    classes = result.boxes.cls.to('cpu').numpy()
    ids = results[0].boxes.id.to('cpu').numpy()
    # convert to dataframe: file_path, x, y, w, h, conf, cls
    result_df = pd.DataFrame({
        'file_path': Path(source).absolute().as_posix(),
        'frame': frame_id,
        'obj_id': ids,
        'x': boxes[:, 0],
        'y': boxes[:, 1],
        'w': boxes[:, 2],
        'h': boxes[:, 3],
        'conf': conf,
        'cls': classes
    })
    result_dfs.append(result_df)

results_df = pd.concat(result_dfs)
results_df.to_parquet(dest / 'tracking.parquet')
results_df.to_csv(dest / 'tracking.csv')
