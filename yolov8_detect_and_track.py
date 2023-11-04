""" Script to run inference on images, videos, or folder/url of images & videos using YOLOv8 """
import argparse
from pathlib import Path

import numpy as np
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

device = 'mps' if torch.backends.mps.is_available() else '0'  # choose mps if available
# inference on source
results = model.track(
    source,
    stream=True,  # avoid memory overflow
    device=device,
    save=plot,  # save plotted results to save_dir
    show_labels=True,
    line_width=3,
    project=save_dir,
    name=dest_name,
    tracker="bytetrack.yaml",
    # tracker="botsort.yaml",
    conf=0.1,
    iou=0.5,
)  # in stream mode, return a generator
# results = list(results)  # convert to list, trigger to run inference

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

    result_np = result.numpy()
    boxes = result_np.boxes.xywh
    conf = result_np.boxes.conf
    classes = result_np.boxes.cls
    ids = result_np.boxes.id
    frames = np.full_like(ids, frame_id)

    # convert to dataframe: file_path, frame, x, y, w, h, conf, cls
    result_df = pd.DataFrame({
        'file_path': Path(source).absolute().as_posix(),
        'frame': frames,
        'obj_id': ids,
        'x': boxes[:, 0],
        'y': boxes[:, 1],
        'w': boxes[:, 2],
        'h': boxes[:, 3],
        'conf': conf,
        'cls': classes
    })
    result_dfs.append(result_df)

dest = Path(args.save_dir) / dest_name
dest.mkdir(parents=True, exist_ok=True)
results_df = pd.concat(result_dfs)
try:
    results_df.to_parquet(dest / 'tracking.parquet')
except ImportError as e:
    print(f"'pyarrow' or 'fastparquet' not found, only saving to csv instead. See error:{e}")
results_df.to_csv(dest / 'tracking.csv')

# convert to mot17 format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
# TODO: refactor the following conversion, more readable
mot_df = (
    results_df
    .copy()
    .loc[:, ['frame', 'obj_id', 'x', 'y', 'w', 'h', 'conf']]
    .rename(columns={'obj_id': 'id',
                     'x': 'bb_left',
                     'y': 'bb_top',
                     'w': 'bb_width',
                     'h': 'bb_height'})
    .sort_values(['frame', 'id'])
)
mot_df['bb_left'] = mot_df['bb_left'] - mot_df['bb_width'] / 2
mot_df['bb_top'] = mot_df['bb_top'] - mot_df['bb_height'] / 2
# add dummy values for x, y, z
mot_df.loc[:, ['x', 'y', 'z']] = -1
# update data type
mot_df.loc[:, ['frame', 'id']] = mot_df.loc[:, ['frame', 'id']].astype(int)
# update frame id to start from 1
mot_df.loc[:, 'frame'] = mot_df.loc[:, 'frame'] + 1
# save to disk
mot_df.to_csv(dest / 'tracking_mot.txt', index=False, header=False)
