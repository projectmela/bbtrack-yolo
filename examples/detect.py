from pathlib import Path

from bbtrack_yolo.BBDetecor import BBDetector, BBDetectorConfig

test_video_path = Path("../tests/data/0190_0.5s.mp4")
test_model_path = Path(
    "../tests/data/d=mc_dtc2023_m=yolov8n_imgsz=1280_bs=16_20231118-014426/weights/best.pt"
)

assert test_video_path.exists(), "test_video_path does not exist"
assert test_model_path.exists(), "test_model_path does not exist"

# create detector config, "model" is required, other fields are optional
detector_config = BBDetectorConfig(model=test_model_path)

# initialize detector
detector = BBDetector(config=detector_config)

# get detections
dets = detector.detect(test_video_path)

# plot predictions on video
dets.plot_on(test_video_path)
