# The file is supposed to start the tracking operation 
import argparse
import os
import glob
from detection_tracking import YoloTracker
import shutil
import time

def argument_parser (parser):

    parser = argparse.ArgumentParser()

    # Detect and track 
    parser.add_argument('-m', '--model', type=str, required=True, help='model.pt path')
    parser.add_argument('-s', '--source', type=str, required=True, help='source to predict')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, default to 1')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader, default to 8')
    parser.add_argument('--save_dir', type=str, default='predictions', help='save directory, default to "predictions"')
    parser.add_argument('--plot', action='store_true', help='save plotted results to save_dir')
    parser.add_argument('--server', action="store_true", help="The data is processed from server", default=False)

    return parser.parse_args()


def find_files(directory, file_format = ".MP4"):
    
    file_paths = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_format):
                # Append the full path to the list
                file_paths.append(os.path.join(root, file))
    
    return file_paths

def print_arguments(args):
    print("Input arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def setup_server_dumping(local_dir_path):
    # if Dicrectory exists, remove it's content and clean it 
    if os.path.exists(local_dir_path):
        files = find_files(local_dir_path, ".*")
        for file in files:
            deleteFile(file)
    else:
        #Make directory if it does not exist
        os.mkdir(local_dir_path)

    return local_dir_path

def copyFile(src, dst) -> str:
    shutil.copy(src,dst)
    file_path = os.path.join(dst, os.path.basename(src))
    return file_path


def deleteFile(dst):
    if os.path.exists(dst):
        print(f"Removing : {dst}")
        os.remove(dst)
    else:
        print(f"File does not exist : {dst}")
        exit()


def clean_up(dir):
        """
        The function will clean up the videos stored after plotting if the clean up function is called.
        The function is a HACK implementation dur to bug in the YOLO processing pipeline. 
        """
        files = glob.glob(os.path.join(dir,"*"))
        for file in files : 
            if file.endswith(".avi"):
                print(f"Removing file :  {file}")
                os.remove(file)
            if os.path.isdir(file):
                 num_of_files = glob.glob(os.path.join(file,"*"))
                 print(f"Removing dir {len(num_of_files)} from {file}. \n Sample file: {num_of_files[0]}")
                 shutil.rmtree(file)

def main():

    # Prepare program
    parser = argparse.ArgumentParser(description="Print an argument")
    args = argument_parser(parser)
    print_arguments(args)

    # Fetch video files from the input directory path for the videos to be processed 
    dir_path = args.source
    if os.path.exists(dir_path):
        files = find_files(dir_path, ".MP4")
    else:
        print("Provided path does not exist")
        exit()

    if not args.plot:
        print("PLOT is not enabled. For batch processing it must be enabled.")
        exit()

    # start time
    start_time_session = time.time()
    
    # If server is to be used then the path has to point to server and we have to copy the data in a local drive before processing the videos. 
    local_dir = setup_server_dumping("dataset/testData")

    # For each file we process it through the detector 
    for file in files: 
        start_time_file = time.time()
        print(f"File being processed: {file}/{len(files)}\n")
        if args.server: # If server is to be used then we want to copy the file first and then process it with YOLO
            local_file_path = copyFile(file,local_dir)
            print(f"File copied to : {local_file_path}")
            if os.path.exists(local_file_path):
                args.source = local_file_path
            else:
                print("Copy function did not work")
                exit()
        else:
            args.source = file
        print(f"Video to be processed :{args.source}")
        
        
        # Run Tracking 
        tracker = YoloTracker(args)
        
        results_dir = tracker.run()

        if args.plot:
            print("\n BUG FIX:")
            clean_up(results_dir)


        # Delete video file from local computer 
        if args.server:
            deleteFile(args.source)

        end_time_file = time.time()
        execution_time_for_file =  end_time_file - start_time_file 
        print(f"Execution time for the {file} = {execution_time_for_file} secs.")

    end_time_session = time.time()
    execution_time_for_session =  end_time_session - start_time_session
    print(f"\nExecution time for the session {dir_path} = {execution_time_for_session} secs.")

if __name__ == "__main__":
    main()