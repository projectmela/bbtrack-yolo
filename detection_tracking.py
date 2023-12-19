""" Script to run inference on images, videos, or folder/url of images & videos using YOLOv8 """
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger
from ultralytics import YOLO

from utility import cur_dt_str


def argument_parser(parser):

    parser.add_argument('-m', '--model', type=str, required=True, help='model.pt path')
    parser.add_argument('-s', '--source', type=str, required=True, help='source to predict')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, default to 1')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
    parser.add_argument('--save_dir', type=str, default='predictions', help='save directory, default to "predictions"')
    parser.add_argument('--plot', action='store_true', help='save plotted results to save_dir')

    return parser.parser_args()

# The class is used for detection and tracking of yolo based on specified model
class YoloTracker: 

    def __init__(self, args ) -> None:

        self.model_file = args.model
        self.model_file = args.model
        self.source = args.source
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.save_dir = args.save_dir
        self.plot = args.plot    

    def process_tracking_results(self, results):
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

            # convert to dataframe: file_path, frame, id, x, y, w, h, conf, cls
            result_df = pd.DataFrame({
                'file_path': Path(self.source).absolute().as_posix(),
                'frame': frames,
                'id': ids,
                'bb_left': boxes[:, 0] - boxes[:, 2] / 2,
                'bb_top': boxes[:, 1] - boxes[:, 3] / 2,
                'bb_width': boxes[:, 2],
                'bb_height': boxes[:, 3],
                'conf': conf,
                'cls': classes,
            })
            result_dfs.append(result_df)
        
        return result_dfs
    

    def save_results(self, results_df, dest_name):
        """The function saves the results as .parquet, .csv and .txt 

        Args:
            results_df (_dataFrame_): Results in data frame format 
            dest_name (_str_): Destination to save results 
        """
     
        # save results
        dest = Path(self.save_dir) / dest_name
        dest.mkdir(parents=True, exist_ok=True)
        print(f'Saving results to {dest.absolute().as_posix()}')
        try:
            results_df.to_parquet(dest / 'tracking.parquet')
        except ImportError as e:
            print(f"'pyarrow' or 'fastparquet' not found, only saving to csv instead. See error:{e}")

        results_df.to_csv(dest / 'tracking.csv')

        # only keep bb class and convert to mot17 format:
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        results_df.loc[:, ['x', 'y', 'z']] = -1 # add dummy values for x, y, z
        results_df.loc[:, ['frame', 'id']] = results_df.loc[:, ['frame', 'id']].astype(int) # update data type

        # remove ids that never classified as blackbuck
        bb_cls_name = ['bb', 'bbfemale', 'bbmale']
        ever_as_bb = results_df.groupby("id").apply(lambda x: any(x["cls_name"].isin(bb_cls_name)))
        results_df = results_df[results_df["id"].isin(ever_as_bb[ever_as_bb].index)]

        # update frame id to start from 1
        results_df.loc[:, 'frame'] = results_df.loc[:, 'frame'] + 1
        (
            results_df
            # keep only mot columns
            .loc[:, ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']]
            .to_csv(dest / 'tracking_blackbuck_mot.txt', index=False, header=False)
        )
        return dest

    def run_yolo_tracking(self):

        model = YOLO(self.model_file)

        # generate string describing model and dataset to name saving directory
        model_name = Path(self.model_file).parent.parent.name
        source_name = Path(self.source).stem
        dest_name = f'{model_name}_{source_name}_{cur_dt_str()}'

        device = 'mps' if torch.backends.mps.is_available() else '0'  # choose mps if available
        # inference on source
        results = model.track(
            self.source,
            stream=True,  # avoid memory overflow
            device=device,
            save=self.plot,  # save plotted results to save_dir
            show_labels=True,
            line_width=3,
            project=self.save_dir,
            name=dest_name,
            tracker="bytetrack.yaml",
            # tracker="botsort.yaml",
            conf=0.1,
            iou=0.5,
        )
        
        result_dfs = self.process_tracking_results(results)

        results_df = pd.concat(result_dfs)
        if model.names is not None:
            results_df.loc[:, 'cls_name'] = results_df.loc[:, 'cls'].apply(lambda x: model.names[x])
        else:
            print('No class names found, using class id as class name')
            results_df.loc[:, 'cls_name'] = results_df.loc[:, 'cls'].astype(str)

        return results_df, dest_name

    def run(self):
        
        # setup model and run tracking 
        results_df, dest_name = self.run_yolo_tracking()

        # Save results and get name of the directory
        dest = self.save_results(results_df, dest_name)

        return dest


# The whole function is converted to an argument and run through a function
def detect_and_track(args):

    model_file = args.model
    source = args.source
    batch_size = args.batch_size # todo : Not used 
    workers = args.workers # todo : Not used 
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
            'id': ids,
            'bb_left': boxes[:, 0] - boxes[:, 2] / 2,
            'bb_top': boxes[:, 1] - boxes[:, 3] / 2,
            'bb_width': boxes[:, 2],
            'bb_height': boxes[:, 3],
            'conf': conf,
            'cls': classes,
        })
        result_dfs.append(result_df)

    results_df = pd.concat(result_dfs)
    if model.names is not None:
        results_df.loc[:, 'cls_name'] = results_df.loc[:, 'cls'].apply(lambda x: model.names[x])
    else:
        print('No class names found, using class id as class name')
        results_df.loc[:, 'cls_name'] = results_df.loc[:, 'cls'].astype(str)

    # save results
    dest = Path(args.save_dir) / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    print(f'Saving results to {dest.absolute().as_posix()}')
    try:
        results_df.to_parquet(dest / 'tracking.parquet')
    except ImportError as e:
        print(f"'pyarrow' or 'fastparquet' not found, only saving to csv instead. See error:{e}")
    results_df.to_csv(dest / 'tracking.csv')

    # only keep bb class and convert to mot17 format:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    results_df.loc[:, ['x', 'y', 'z']] = -1 # add dummy values for x, y, z
    results_df.loc[:, ['frame', 'id']] = results_df.loc[:, ['frame', 'id']].astype(int) # update data type

    # remove ids that never classified as blackbuck
    bb_cls_name = ['bb', 'bbfemale', 'bbmale']
    ever_as_bb = results_df.groupby("id").apply(lambda x: any(x["cls_name"].isin(bb_cls_name)))
    results_df = results_df[results_df["id"].isin(ever_as_bb[ever_as_bb].index)]

    # update frame id to start from 1
    results_df.loc[:, 'frame'] = results_df.loc[:, 'frame'] + 1
    (
        results_df
        # keep only mot columns
        .loc[:, ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']]
        .to_csv(dest / 'tracking_blackbuck_mot.txt', index=False, header=False)
    )

# This function will allow anyone to run the file
def main(args):

    parser = argparse.ArgumentParser()
    args = argument_parser (parser)

    # Idea 1 : 
    detect_and_track(args)
    
    # Option 2 : 
    yolo_tracker = YoloTracker(args)
    yolo_tracker.run()

if __name__ == "__main__":
    
    main()