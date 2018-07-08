import sys
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    exit(1)

import numpy as np
import argparse
import cv2
import pyrealsense2 as rs

sys.path.append('..')
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
#    net = cv2.dnn.readNetFromTensorflow('/home/b920405/git/MobileNet-SSD-RealSense-TF/frozen_inference_graph.pb',
#                                        '/home/b920405/git/MobileNet-SSD-RealSense-TF/graph.pbtxt')

    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

    net = cv2.dnn.readNetFromTensorflow('ssdlite_mobilenet_v2_coco.pb')
    swapRB = True
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

        blob = cv2.dnn.blobFromImage(color_image, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()

        height = color_image.shape[0]
        width = color_image.shape[1]

        num_valid_boxes = int(detections.shape[2])

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
