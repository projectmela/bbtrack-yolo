# Blackbuck Track YOLO

Using YOLOv8 to detect and track blackbucks in videos.

## File structure

- [dir] **dataset**: dataset to train or to predict
- [dir] **models**: trained models and related files like configurations
    - [dir] **{dataset_name}\_{train_params}\_{datetime}**: model folder
        - [dir] **weights**: weights files ends with `.pt`
- [dir] **predictions**: prediction results
    - [dir] **{model_params}\_{data_source_name}\_{datetime}**: predictions folder
- [dir] **environment**: conda environment files
- [dir] **utility**: package of utility functions

## Set up local environment

```bash
# for macOS with apple silicon
conda env create -f env/apple_mps.yaml
# for x86 linux with nvidia gpu
conda env create -f env/linux_cuda.yaml
```

YOLOv8 docker image is available at [Docker Hub: ultralytics/ultralytics](https://hub.docker.com/r/ultralytics/ultralytics).

## Train

```bash
python3 yolov8_train.py \
  --dataset path/to/dataset/.yaml
# or specify more parameters see:
python3 yolov8_train.py --help
``` 

## Predict (Detection and Tracking)

```bash
# see script argument detail
python3 yolov8_detect_and_track.py -h
```