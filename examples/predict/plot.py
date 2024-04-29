from pathlib import Path

from bbtrack_yolo.BBoxDetection import BBoxDetection

# find all prediction files: "predictions/{model_name}/{sequence_name}/track.csv"
prediction_files = list(Path("predictions").glob("*/*/tracks.csv"))
print("Found prediction files:")
print("\n".join([str(pf) for pf in prediction_files]))

# find all videos
video_files = list(Path("videos").glob("*.[mM][pP]4"))
print("Found video files:")
print("\n".join([str(vf) for vf in video_files]))

# match predictions to videos
video_pred_pairs = []
for vf in video_files:
    for pf in prediction_files:
        if any(vf.stem in part for part in pf.parts):
            video_pred_pairs.append((vf, pf))
            break
    else:
        print(f"No match for {vf}")

print("Matched video-prediction pairs:")
print("\n".join([f"{str(vf)} -> {str(pf)}" for vf, pf in video_pred_pairs]))

# load tracks and plot
for vf, pf in video_pred_pairs:
    BBoxDetection.load_from(pf).plot_on(vf)
