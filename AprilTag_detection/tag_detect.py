import cv2
from dt_apriltags import Detector
from math import atan2, cos, asin, pi, isclose

# Calibrate your camera to get the following parameters
FX, FY, CX, CY = (1844.8698633912334, 1722.3078724323725, 1248.519572471147, 942.224758060514)

# If another tag family is used, change it here
detector = Detector(families="tag25h9", nthreads=4, quad_decimate=1.0, decode_sharpening=0.5)
camera = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not camera.isOpened():
    raise IOError("Cannot open camera")

while True:
    success, frame = camera.read()

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(grayImage, estimate_tag_pose=True, camera_params=[FX, FY, CX, CY], tag_size=0.087) # tag_size=0.206
    print("[INFO] {} total AprilTags detected".format(len(detections)))

    # loop over the AprilTag detection results
    for tag in detections:

        R = tag.pose_R

        # Calculation for the euler angles
        # Source: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
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

        print(tag.pose_t[2])

        (ptA, ptB, ptC, ptD) = tag.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = tag.tag_family.decode("utf-8")
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily))
    # show the output image after AprilTag detection
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()