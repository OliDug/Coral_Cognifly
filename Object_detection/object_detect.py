from aiymakerkit import vision
from pycoral.utils.dataset import read_label_file
import os.path

def path(name):
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, name)

OBJECT_DETECTION_MODEL = path('ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite')
LABEL_FILE = path('coco_labels.txt')

CAMERA_INDEX = 1

detector = vision.Detector(OBJECT_DETECTION_MODEL)
labels = read_label_file(LABEL_FILE)

for frame in vision.get_frames(capture_device_index=CAMERA_INDEX, display=True):
    objects = detector.get_objects(frame, threshold=0.3)
    for object in objects:
        # Remove the if to draw all the detected objects
        if object.id == 43: # bottle
            vision.draw_objects(frame, [object], labels)