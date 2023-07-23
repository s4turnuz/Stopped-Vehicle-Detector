# Stopped Vehicle Detection with YOLOv8 and SORT

This is a Python script for performing Stopped Vehicle Detection using YOLOv8 (You Only Look Once version 8) for object detection and SORT (Simple Online and Realtime Tracking) for tracking the detected objects.

<img src="src/def.gif" alt="Results">

## Requirements

To run this script, you need to have the following installed:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository containing this script and other required files.
2. Make sure you have all the dependencies installed.
3. Run the script with the required command-line arguments.

## Command-line Arguments

The script accepts the following command-line arguments:

1. `--input`: Path to the input video file.
2. `--output`: Path to the output video file.
3. `--seconds`: Seconds to consider a vehicle as stopped.
4. `--yolo_weights`: Path to the YOLOv8 model weights file.

Example usage:

```bash
python script_name.py --input input_video.mp4 --output output_video.mp4 --seconds 5 --yolo_weights yolo_weights.pth
```

## Description

The script performs the following steps:

1. Parses the command-line arguments for input video file, output video file, stopped vehicle time threshold, and YOLO model weights file.
2. Initializes the necessary variables and video capture.
3. Reads frames from the input video.
4. Performs object detection using YOLOv8 on each frame to detect vehicles.
5. Applies SORT (Simple Online and Realtime Tracking) to track the detected vehicles across frames.
6. Identifies stopped vehicles by calculating the time they have been in the same position and comparing it to the threshold.
7. Annotates the stopped vehicles with rectangles and labels in the output video.
8. Saves the annotated output video.

The script uses YOLOv8 for object detection and SORT for vehicle tracking. It labels and tracks vehicles of certain classes (e.g., "car," "truck," "bus," "motorbike") with a confidence threshold of 0.3.

Make sure you have the necessary YOLO model weights file for accurate detection results.

## Note

Please ensure that you have proper licenses for the YOLOv8 model, as it might be subject to specific usage terms and conditions.

This script is designed for stopped vehicle detection and can be further customized and extended for other applications as needed.

Feel free to reach out if you have any questions or need further assistance. Happy coding!