# The file is supposed to start the tracking operation 
import argparse
import os
import glob
from detection_tracking import detect_and_track

def argument_parser ():
    parser = argparse.ArgumentParser()

    # Training 
    """     
    parser.add_argument('-m', '--model', type=str, default='yolov8l.pt', help='Model.pt path(s) or pretrained model name')
    parser.add_argument('-d', '--dataset', type=str, default='dataset/bb_2022/bb_2022.yaml', help='dataset.yaml path')
    parser.add_argument('--image_size', type=int, default=5472,
                        help='image size, default to original size in blackbuck dataset, 5472')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size, default to auto (-1)')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs, default to 10')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience, default to 100') 
    """

    # Detect and track 
    parser.add_argument('-m', '--model', type=str, required=True, help='model.pt path')
    parser.add_argument('-s', '--source', type=str, required=True, help='source to predict')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, default to 1')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
    parser.add_argument('--save_dir', type=str, default='predictions', help='save directory, default to "predictions"')
    parser.add_argument('--plot', action='store_true', help='save plotted results to save_dir')

    return parser.parse_args()

def glob_files(path, file_format = "*.MP4"):
    # Construct the pattern to match files
    pattern = os.path.join(path, file_format)

    # Use glob to find files matching the pattern
    files = glob.glob(pattern)

    return files




def main():
    parser = argparse.ArgumentParser(description="Print an argument")
    args = argument_parser(parser)

    dir_path = args.source
    if os.path.exists(dir_path):
        files = glob_files(dir_path)

    else:
        print("Provided path does not exist")
        exit()

    # Changing the input datatype to run the required affair
    for file in files: 
        args.source = file
        detect_and_track(args)


if __name__ == "__main__":
    main()