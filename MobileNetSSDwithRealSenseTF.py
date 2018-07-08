import sys
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    exit(1)

import os
import numpy as np
import argparse
import cv2
import pyrealsense2 as rs
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append('..')
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

try:
    CWD_PATH = os.getcwd()
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')
    swapRB = True
    NUM_CLASSES = 90
    classNames = { 0: 'background',
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
        7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
        13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
        18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
        24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
        32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
        37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
        41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
        46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }

    ## Load the label map.
    # Label maps map indices to category names, so that when the convolution
    # network predicts `5`, we know that this corresponds to `airplane`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Configure depth and color streams RealSense D435
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        frame_expanded = np.expand_dims(color_image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            color_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1,
            min_score_thresh=0.85)

        #https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

        #meters = depth_frame.as_depth_frame().get_distance(x1+int((x2-x1)/2), y1+int((y2-y1)/2))
        #label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"+ " {:.2f}".format(meters) + " meters away"
        #cv2.putText(color_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)





        if num_valid_boxes > 0:

            for box_index in range(num_valid_boxes):
                
                if (not np.isfinite(detections[0, 0, i, 0]) or
                    not np.isfinite(detections[0, 0, i, 1]) or
                    not np.isfinite(detections[0, 0, i, 2]) or
                    not np.isfinite(detections[0, 0, i, 3]) or
                    not np.isfinite(detections[0, 0, i, 4]) or
                    not np.isfinite(detections[0, 0, i, 5]) or
                    not np.isfinite(detections[0, 0, i, 6])):
                    continue

                x1 = max(0, int(detections[0, 0, i, 3] * height))
                y1 = max(0, int(detections[0, 0, i, 6] * width))
                x2 = min(height, int(detections[0, 0, i, 5] * height))
                y2 = min(width, int(detections[0, 0, i, 4] * width))

                min_score_percent = 60
                class_id = int(detections[0, 0, i, 1])
                percentage = int(detections[0, 0, i, 2] * 100)
                if (percentage <= min_score_percent):
                    continue

                meters = depth_frame.as_depth_frame().get_distance(x1+int((x2-x1)/2), y1+int((y2-y1)/2))
                #print(meters)
                label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"+ " {:.2f}".format(meters) + " meters away"

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, box_thickness)

                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)

                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = x1
                label_top = y1 - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(color_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(color_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', cv2.resize(color_image,(width, height)))
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

except:
    import traceback
    traceback.print_exc()

finally:

    # Stop streaming
    pipeline.stop()
    print("\n\nFinished\n\n")
    sys.exit()
