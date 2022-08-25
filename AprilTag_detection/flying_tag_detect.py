import cv2
from dt_apriltags import Detector
from cognifly import Cognifly
from cognifly.utils.pid import PID
import time
from math import atan2, cos, asin, pi, isclose

# Size of the frames returned by the detector
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Distance from the tag that the Cognifly should aim for
PROXIMITY_GOAL = 3.0

# Tolerance on the yaw angle the Cognifly aims for
MIN_YAW_ANGLE = -1.0
MAX_YAW_ANGLE = 0.07

MAX_YAW_VELOCITY = 0.2
MAX_VELOCITY = 0.5

# Camera parameters from calibration
FX, FY, CX, CY = (1877.0244059141842, 1740.9506161468948, 1241.6701604190541, 973.2745910050869)

# If another tag family is used, change it here
detector = Detector(families="tag25h9", nthreads=4, quad_decimate=1.0, decode_sharpening=0.5)
camera = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not camera.isOpened():
    raise IOError("Cannot open camera")

# If PIDs don't behave properly, tune the gains here
# Forward and backward
xPid = PID(kp=0.05, ki=0.0, kd=0.0001, setpoint=PROXIMITY_GOAL,
        sample_time=None, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))
# Right and left
yPid = PID(kp=0.001, ki=0.0, kd=0.001, setpoint=FRAME_WIDTH/2,
        sample_time=None, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))
# Up and down
zPid = PID(kp=0.005, ki=0.0, kd=0.0, setpoint=FRAME_HEIGHT/2,
        sample_time=None, output_limits=(-MAX_VELOCITY, MAX_VELOCITY))
# Yaw control
yawPid = PID(kp=2.0, ki=1.0, kd=2.0, setpoint=0.0,
        sample_time=None, output_limits=(-MAX_YAW_VELOCITY, MAX_YAW_VELOCITY))

drone = Cognifly(drone_hostname="192.168.5.247", gui=False)
drone.arm()
time.sleep(5.0)
drone.takeoff_nonblocking()
time.sleep(2.0)

landed = False
while not landed:
    success, frame = camera.read()

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(grayImage, estimate_tag_pose=True, camera_params=[FX, FY, CX, CY], tag_size=0.203) #tag_size=0.087
    
    for tag in tags:

        if tag.tag_id != 0:
            continue
        
        if (abs(tag.center[0]) - FRAME_WIDTH/2 <= 15 and 
           abs(tag.center[1]) - FRAME_HEIGHT/2 <= 15 and 
           isclose(tag.pose_t[2], PROXIMITY_GOAL, rel_tol= 0.2)):
                # Sequence for landing on the base station. Tune the duration of the forward acceleration or 
                # the acceleration if the drone has the time to raise above the base station.
                drone.set_velocity_nonblocking(v_x=-3.0, v_y=0.0, v_z=0.0, w=0.0, duration=1.4, drone_frame=True)
                time.sleep(1.4)
                drone.land_nonblocking()
                time.sleep(1.0)
                drone.disarm()
                landed = True
                break

        R = tag.pose_R
        # Calculation for the euler angles
        #Source: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
        if not isclose(R[2,0], 1) and not isclose(R[2,0], -1):
            tagPitch = -asin(R[2,0])
            tagYaw = atan2(R[2,1]/cos(tagPitch), R[2,2]/cos(tagPitch))
            
            tagRoll = atan2(R[1,0]/cos(tagPitch), R[0,0]/cos(tagPitch))
        else:
            tagRoll = 0
            if isclose(R[2,0], -1):
                tagPitch = pi/2
                tagYaw = tagRoll + atan2(R[0,1], R[0,2])
            else:
                tagPitch = -pi/2
                tagYaw = -tagRoll + atan2(-R[0,1], -R[0,2])

        tolerance = abs(MIN_YAW_ANGLE) if tagYaw < 0 else MAX_YAW_ANGLE
        if not isclose(tagYaw, 0.0, rel_tol=tolerance):
                yawVelocity = yawPid(input_=tagYaw)

        xVelocity = xPid(input_=tag.pose_t[2])
        yVelocity = yPid(input_=tag.center[0])
        zVelocity = zPid(input_=tag.center[1])

        drone.set_velocity_nonblocking(v_x=xVelocity, v_y=yVelocity, v_z=zVelocity, w=yawVelocity, duration=1.0, drone_frame=True)