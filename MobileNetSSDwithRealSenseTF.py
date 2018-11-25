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
sys.path.append('..')
from utils import label_map_util
import visualization_utils as vis_util

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
pipeline = None

try:
    HOME_PATH = os.path.expanduser('~')
    CWD_PATH = 'model'
    #MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'mscoco_label_map.pbtxt')
    swapRB = True
    NUM_CLASSES = 90

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
except:
    import traceback
    traceback.print_exc()
    # Stop streaming
    if pipeline != None:
        pipeline.stop()
    print("\n\nFinished\n\n")
    sys.exit()


def init():
    glClearColor(0.7, 0.7, 0.7, 0.7)

def idle():
    glutPostRedisplay()

def resizeview(w, h):
    glViewport(0, 0, w, h)
    glLoadIdentity()
    glOrtho(-w / 640, w / 640, -h / 480, h / 480, -1.0, 1.0)

def keyboard(key, x, y):
    key = key.decode('utf-8')
    if key == 'q':
        sys.exit()

def camThread():
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return
    
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    height = color_image.shape[0]
    width = color_image.shape[1]

    frame_expanded = np.expand_dims(color_image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    img = vis_util.visualize_boxes_and_labels_on_image_array(
        color_image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.55,
        depth_frame=depth_frame,
        height=height,
        width=width)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor3f(1.0, 1.0, 1.0)
    glEnable(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBegin(GL_QUADS) 
    glTexCoord2d(0.0, 1.0)
    glVertex3d(-1.0, -1.0,  0.0)
    glTexCoord2d(1.0, 1.0)
    glVertex3d( 1.0, -1.0,  0.0)
    glTexCoord2d(1.0, 0.0)
    glVertex3d( 1.0,  1.0,  0.0)
    glTexCoord2d(0.0, 0.0)
    glVertex3d(-1.0,  1.0,  0.0)
    glEnd()
    glFlush()
    glutSwapBuffers()

try:
    glutInitWindowPosition(0, 0)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE )
    glutCreateWindow("DEMO")
    glutFullScreen()
    glutDisplayFunc(camThread)
    glutReshapeFunc(resizeview)
    glutKeyboardFunc(keyboard)
    init()
    glutIdleFunc(idle)
    glutMainLoop()
except:
    import traceback
    traceback.print_exc()
finally:
    # Stop streaming
    if pipeline != None:
        pipeline.stop()
    print("\n\nFinished\n\n")
    sys.exit()
