from pathlib import Path

from bbtrack_yolo.BBDetecor import BBDetector, BBDetectorConfig

model_path = Path("nano_bb.pt")

video_dir = Path("videos")
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
    dets, dets_path = detector.detect(vp)

    # optional: do tracking with detections;
    # suggestion: do tracking in parallel later with batch_track.py
    # tracker = BBTracker(config=BYTETrackerConfig())
    # trks = tracker.track(dets)
    # trks.save_to(dets_path.parent / "tracks.csv")

    # plot predictions on video
    # dets.plot_on(vp)
    # trks.plot_on(vp)
