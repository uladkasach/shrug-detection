#Import the OpenCV library
import cv2
import argparse
import time
import numpy as np;
import traceback
from scipy import stats;
import time;
import datetime;

## parse arguments
parser = argparse.ArgumentParser();
parser.add_argument('--cascade_path', help='path to cascade xml', default='/usr/local/lib/python2.7/dist-packages/cv2/data/haarcascade_frontalface_default.xml');
parser.add_argument('--video', help='path to video, defaults to webcam feed', default=0)
args = parser.parse_args();



############################################################
def find_max_face_in_faces(faces):
    # find max face by area (w*h)
    max_area = 0
    max_face = None;
    for (x,y,w,h) in faces:
        if  w*h > max_area:
            max_area = w*h
            max_face = (x,y,w,h);
    return max_face;
def detect_face(img_gray):
    # use cascade detector to find all faces in image
    faces = face_cascade_classifier.detectMultiScale(img_gray, 1.3, 5)

    # extract max face from faces
    face = find_max_face_in_faces(faces);

    # return the face
    return face;
def draw_rectange_onto_image(img, rectangle):
    (x,y,w,h) = rectangle;
    cv2.rectangle(
        img,
        (x-10, y-20),
        (x + w+10 , y + h+20),
        rectangle_color,
        2)
    return img;
def draw_lines_onto_image(img, lines, color="GREEN"):
    # generate start, end for each line
    line_pairs = [];
    for index in range(len(lines)):
        if(index == 0): continue;
        start = lines[index-1];
        end = lines[index];
        line_pairs.append((start, end));

    # draw each line onto image
    for line_pair in line_pairs:
        img = draw_line_onto_image(img, line_pair, color=color);

    # return image
    return img;
def draw_line_onto_image(img, line, color="GREEN"):
    beginning = line[0];
    end = line[1];
    if(color=="GREEN"): color = (0, 255, 0);
    if(color=="BLUE"): color = (255, 0, 0);
    if(color=="RED"): color = (0, 0, 255);
    cv2.line(img, beginning, end, color, 2)
    return img;
def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=3, search_width = 20):
    columns = img_gray[y:y+h, x-search_width/2:x+search_width/2];
    column_average = columns.mean(axis=1);
    gradient = np.gradient(column_average, 3);
    gradient = np.absolute(gradient); # abs gradient value
    max_indicies = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:] # indicies of the top 5 values
    max_values = gradient[max_indicies];
    if(max_values.sum() < top_values_to_consider): return None; # return none if no large gradient exists - probably no shoulder in the range
    weighted_indicies = (max_indicies * max_values);
    weighted_average_index = weighted_indicies.sum() / max_values.sum();
    index = int(weighted_average_index);
    index = y + index;
    return index;
def detect_shoulder(img_gray, face, direction, x_scale=0.75, y_scale=0.75):
    x_face, y_face, w_face, h_face = face; # define face components

    # define shoulder box componenets
    w = int(x_scale * w_face);
    h = int(y_scale * h_face);
    y = y_face + h_face * 3/4; # half way down head position
    if(direction == "right"): x = x_face + w_face - w / 20; # right end of the face box
    if(direction == "left"): x = x_face - w + w/20; # w to the left of the start of face box
    rectangle = (x, y, w, h);

    # calculate position of shoulder in each x strip
    x_positions = [];
    y_positions = [];
    for delta_x in range(w):
        this_x = x + delta_x;
        this_y = calculate_max_contrast_pixel(img_gray, this_x, y, h);
        if(this_y is None): continue; # dont add if no clear best value
        x_positions.append(this_x);
        y_positions.append(this_y);

    # extract line from positions
    #line = [(x_positions[5], y_positions[5]), (x_positions[-10], y_positions[-10])];
    lines = [];
    for index in range(len(x_positions)):
        lines.append((x_positions[index], y_positions[index]));

    # extract line of best fit from lines
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions,y_positions)
    line_y0 = int(x_positions[0] * slope + intercept)
    line_y1 = int(x_positions[-1] * slope + intercept);
    line = [(x_positions[0], line_y0), (x_positions[-1], line_y1)];

    # decide on value
    #value = intercept;
    value = np.array([line[0][1], line[1][1]]).mean();

    # return rectangle and positions
    return line, lines, rectangle, value;



history_dict = dict({
    "RIGHT" : [],
    "LEFT" : [],
});
def update_shoulder_history(history_key, new_value, queue_size=5):
    global history_dict;
    history = history_dict[history_key];
    if(len(history) > queue_size-1): history = history[-queue_size-1:]; # last queue_size-1 (e.g., if queue_size is 5 get last 4)
    history.append(new_value);
    history_dict[history_key] = history;
def determine_if_shrugging(history_key, new_value, queue_size = 5, threshold = 6):
    history = (history_dict[history_key]);
    history.append(new_value);
    history = np.array(history);
    if(len(history) < queue_size): return False; # if not enough history to compute, move on
    stdev = history.std();
    mean = history.mean();
    print(stdev);
    return stdev > threshold;

def find_and_display(capture, writer):
    #Retrieve the latest image from the webcam
    rc,img = capture.read()

    # detection expects grayscale image, convert to grayscale to run basic algorithm of detecting shoulder
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        # extract the face
        face = detect_face(img_gray);

        # detect shoulders
        right_shoulder_line, right_shoulder_lines, right_shoulder_rectangle, right_shoulder_value = detect_shoulder(img_gray, face, "right")
        left_shoulder_line, left_shoulder_lines, left_shoulder_rectangle, left_shoulder_value = detect_shoulder(img_gray, face, "left")

        # update shoulder histories and determine if shrugging
        queue_size = 10;
        right_shrugging = determine_if_shrugging("RIGHT", right_shoulder_value, queue_size=queue_size);
        update_shoulder_history("RIGHT", right_shoulder_value, queue_size=queue_size);
        if(right_shrugging):
            right_color = "RED";
        else:
            right_color = "GREEN";
        left_shrugging = determine_if_shrugging("LEFT", left_shoulder_value, queue_size=queue_size);
        update_shoulder_history("LEFT", left_shoulder_value, queue_size=queue_size);
        if(left_shrugging):
            left_color = "RED";
        else:
            left_color = "GREEN";

        # plot face onto image
        img = draw_rectange_onto_image(img, face);
        img = draw_rectange_onto_image(img, right_shoulder_rectangle);
        img = draw_lines_onto_image(img, right_shoulder_lines, color="BLUE");
        img = draw_line_onto_image(img, right_shoulder_line, color=right_color);
        img = draw_rectange_onto_image(img, left_shoulder_rectangle);
        img = draw_lines_onto_image(img, left_shoulder_lines, color="BLUE");
        img = draw_line_onto_image(img, left_shoulder_line, color=left_color);
    except Exception as e:
        print(e);
        print(traceback.format_exc())
        #print("face not found!");
        img = img;

    # draw the face
    cv2.imshow('webcam_feed', img)

    # record frame
    # print(img.shape)
    # img = np.random.randint(255, size=img.shape).astype('uint8')
    writer.write(img);

    # wait so that image has time to draw
    cv2.waitKey(50) # wait for up to N milliseconds


#################
## init globals
#################
#Initialize a face cascade classifier
face_cascade_classifier = cv2.CascadeClassifier(args.cascade_path)

# define rectangle colro
rectangle_color = (0,165,255)

#Open the first video device
capture = cv2.VideoCapture(args.video)
frame_width = int( capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int( capture.get( cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_height, frame_width);
if(args.video == 0):
    ts = time.time();
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    video_name = "webcam_output_" + st;
else:
    video_name = str(args.video);
writer = cv2.VideoWriter(video_name + "_labeled.avi", cv2.VideoWriter_fourcc(*'PIM1'), 25, (frame_width, frame_height))

while True:
    find_and_display(capture, writer);
