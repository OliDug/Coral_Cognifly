from aiymakerkit import vision
from pycoral.utils.dataset import read_label_file
from cognifly import Cognifly
from cognifly.utils.pid import PID
import os.path
import time
import math

def path(name):
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, name)

OBJECT_DETECTION_MODEL = path('ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite')
LABEL_FILE = path('coco_labels.txt')

CAMERA_INDEX = 1

# Size of the frames returned by the detector
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Size of the bounding box that indicates the drone is close enough
BBOX_MIN_WIDTH = 110
BBOX_MAX_WIDTH = 150
BBOX_MIN_HEIGHT = 430
BBOX_MAX_HEIGHT = 480

MAX_VELOCITY = 0.7

detector = vision.Detector(OBJECT_DETECTION_MODEL)
labels = read_label_file(LABEL_FILE)

# If PIDs don't behave properly, tune the gains here
# Forward and backward
xPid = PID(kp=0.005, ki=0.0, kd=0.0001, setpoint=(BBOX_MAX_WIDTH+BBOX_MIN_WIDTH)/2,
        sample_time=None, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))
# Right and left
yPid = PID(kp=0.001, ki=0.0, kd=0.0001, setpoint=FRAME_WIDTH/2,
        sample_time=None, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))
# Up and down
zPid = PID(kp=0.001, ki=0.0, kd=0.0, setpoint=FRAME_HEIGHT/2,
        sample_time=None, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))

drone = Cognifly(drone_hostname="192.168.5.247", gui=False)
drone.arm()
time.sleep(5.0)
drone.takeoff_nonblocking()
time.sleep(2.0)

for frame in vision.get_frames(capture_device_index=CAMERA_INDEX, display=False):
    objects = detector.get_objects(frame, threshold=0.3)
    vision.draw_objects(frame, objects, labels)

    for object in objects:

        # Detects a bottle
        if object.id != 43:
            continue

        hBboxCenter = object.bbox.xmax - (object.bbox.width / 2)
        vBboxCenter = object.bbox.ymax - (object.bbox.height / 2)

        if math.isclose(hBboxCenter, FRAME_WIDTH/2, rel_tol=10) and math.isclose(vBboxCenter, FRAME_HEIGHT/2, rel_tol=10) and object.bbox.width < BBOX_MAX_WIDTH and object.bbox.width > BBOX_MIN_WIDTH:
            drone.land_nonblocking()
            time.sleep(3.0)
            drone.disarm()
            break

        xVelocity = -xPid(input_=object.bbox.width)
        yVelocity = -yPid(input_=hBboxCenter)
        zVelocity = zPid(input_=vBboxCenter)

        drone.set_velocity_nonblocking(v_x=xVelocity, v_y=yVelocity, v_z=zVelocity, w=0.0, duration=1.0, drone_frame=True)

