from pathlib import Path

from bbtrack_yolo.BBDetecor import BBDetector, BBDetectorConfig

model_path = Path(
    "/abyss/home/localcode/bb-yolo/"
    "models/models_20231219_ontology/"
    "d=mc_dtc2023_gd-shadow_o=gd-shadow_m=yolov8m_imgsz=5472_bs=1_"
    "20231219-054003/"
    "weights/best.pt"
)

video_dir = Path("data/videos")
video_paths = list(video_dir.glob("*.[mM][pP]4"))

print("Found video files:")
print("\n".join([str(vp) for vp in video_paths]))

# create detector config, "model" is required, other fields are optional
detector_config = BBDetectorConfig(model=model_path)
print(f"{detector_config=}")

# initialize detector
detector = BBDetector(config=detector_config)
print(f"{detector=}")

# iterate over video files
for vp in video_paths:
    print(f"processing {vp=}")

    # get detections
    dets = detector.detect(vp)

    # plot predictions on video
    # dets.plot_on(vp)
