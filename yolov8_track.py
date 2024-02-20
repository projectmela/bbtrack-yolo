""" This is a file to do tracking using yolov8. """

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import trange
from ultralytics.trackers.byte_tracker import BYTETracker

parser = argparse.ArgumentParser()
parser.add_argument('--detection', type=str, required=True,
                    help="a parquet or csv file containing columns: [frame bb_left, bb_top, bb_width, bb_height, conf, cls]")
parser.add_argument('--cls', type=int, required=False, default=None, nargs='+',
                    help="classes to track, in the format of '0,1,...', default to track all classes")


@dataclass(frozen=True)
class BYTETrackerArgs:
    """ Arguments for BYTETracker implementation in YOLOv8 """
    track_high_thresh: float = 0.1  # threshold for the first association
    track_low_thresh: float = 0.01  # threshold for the second association
    new_track_thresh: float = 0.1  # threshold for init new track if the detection does not match any tracks
    track_buffer: int = 30  # buffer to calculate the time when to remove tracks
    match_thresh: float = 0.8  # threshold for matching tracks
    # min_box_area: 10  # threshold for min box areas(for tracker evaluation, not used for now)
    # mot20: False  # for tracker evaluation(not used for now)


@dataclass(frozen=True)
class Detection:
    """ Detections in a single frame
    BYTETracker implementation in YOLOv8 expect the detection object in the following format:
    """
    xywh: np.ndarray  # shape (N, 4), (center x, center y, width, height) of the detections
    conf: np.ndarray  # shape (N,), confidence of the detections
    cls: np.ndarray  # shape (N,), class of the detections


args = parser.parse_args()
det_path = Path(args.detection)
cls = args.cls
assert det_path.exists(), f'Detection file {det_path} does not exist'

det_result = pd.read_parquet(det_path) if det_path.suffix == '.parquet' else pd.read_csv(det_path)

# filter classes
if cls is not None:
    det_result = det_result.query('cls in @cls')

# initialize tracker
tracker = BYTETracker(args=BYTETrackerArgs(), frame_rate=30)

# tracking
min_frame = det_result['frame'].min()
max_frame = det_result['frame'].max()
tracking_results = np.empty((0, 9))
tqdm_desc = "Tracking"
for frame_idx in trange(min_frame, max_frame + 1, desc=tqdm_desc, bar_format='{l_bar}{bar:10}{r_bar}'):

    frame_dets = det_result.query('frame == @frame_idx')

    if frame_dets.empty:
        continue

    frame_dets = frame_dets[['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'cls']].values

    # convert from (left, top, width, height) to (center_x, center_y, width, height)
    frame_dets[:, 0] += frame_dets[:, 2] / 2
    frame_dets[:, 1] += frame_dets[:, 3] / 2

    # intialize Detections
    dets = Detection(
        xywh=frame_dets[:, :4],
        conf=frame_dets[:, 4],
        cls=frame_dets[:, 5],
    )

    # update tracker and obtain tracks (shape (N, 6), (x1, y1, x2, y2, track_id, conf, cls, det_idx_in_frame))
    tracks = tracker.update(results=dets)  # type: np.ndarray

    if tracks.size == 0:
        continue

    # concatenate the frame index to the tracks
    tracks = np.concatenate([np.full((tracks.shape[0], 1), frame_idx), tracks], axis=1)

    tracking_results = np.concatenate([tracking_results, tracks], axis=0)

# convert from ltrb (x1, y1, x2, y2) to ltwh (left, top, width, height)
tracking_results[:, 3] -= tracking_results[:, 1]
tracking_results[:, 4] -= tracking_results[:, 2]

# save tracking results
tracking_df = pd.DataFrame(
    tracking_results,
    columns=['frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'track_id', 'conf', 'cls', 'det_idx_in_frame']
)
tracking_df.to_csv(det_path.parent / 'tracking.csv', index=False)
tracking_df.to_parquet(det_path.parent / 'tracking.parquet', index=False)

# convert to mot format
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
mot_df = tracking_df[['frame', 'track_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']].copy()
mot_df['x'] = 0
mot_df['y'] = 0
mot_df['z'] = 0
mot_df.to_csv(det_path.parent / "tracking_mot.txt", index=False, header=False)
