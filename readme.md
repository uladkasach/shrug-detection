# shrug-detection

This project utilizes OpenCV to detect and track both a persons face and shoulders. Shoulder tracking is then used to heuristically determine whether the user is shrugging or not.

![shrug_demo](https://user-images.githubusercontent.com/10381896/39088739-6ca56fec-4585-11e8-84c3-9873d3f7ab89.png)


## installation
To use this project please install the `cv2`, `numpy`, `scipy`, and `argparse` packages for python (e.g., with `pip`).

## usage
This package can be run with the webcam or with a pre-recorded video.
- webcam usage: `python detect_and_track.py`
- video usage: `python detect_and_track.py --video path_to_file.mp4`
    - e.g., `python detect_and_track.py --video shrug_edited.mp4` - to run the example video

## references
- https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/
