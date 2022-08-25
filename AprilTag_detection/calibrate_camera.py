from __future__ import print_function
import numpy as np
import cv2
import argparse

# In AprilTag_test/Calibration_Images :
# python3 ../calibrate_camera.py img0000.jpg img0001.jpg img0002.jpg img0003.jpg img0004.jpg img0005.jpg img0006.jpg img0007.jpg img0008.jpg img0009.jpg img0010.jpg
# img0011.jpg img0012.jpg img0013.jpg img0014.jpg img0015.jpg img0016.jpg img0017.jpg img0018.jpg img0019.jpg img0020.jpg 
# img0021.jpg img0022.jpg img0023.jpg img0024.jpg img0025.jpg img0026.jpg img0027.jpg img0028.jpg img0029.jpg img0030.jpg 
# -r 7 -c 9 -s 0.257 

def main():

    parser = argparse.ArgumentParser(
        description='calibrate camera intrinsics using OpenCV')

    parser.add_argument('filenames', metavar='IMAGE', nargs='+',
                        help='input image files')

    parser.add_argument('-r', '--rows', metavar='N', type=int,
                        required=True,
                        help='# of chessboard corners in vertical direction')

    parser.add_argument('-c', '--cols', metavar='N', type=int,
                        required=True,
                        help='# of chessboard corners in horizontal direction')

    parser.add_argument('-s', '--size', metavar='NUM', type=float, default=1.0,
                        help='chessboard square size in user-chosen units (should not affect results)')

    parser.add_argument('-d', '--show-detections', action='store_true',
                        help='show detections in window')

    options = parser.parse_args()

    if options.rows < options.cols:
        patternsize = (options.cols, options.rows)
    else:
        patternsize = (options.rows, options.cols)

    sz = options.size

    x = np.arange(patternsize[0])*sz
    y = np.arange(patternsize[1])*sz

    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros_like(xgrid)
    opoints = np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 1, 3)).astype(np.float32)

    imagesize = None

    win = 'Calibrate'
    cv2.namedWindow(win)

    ipoints = []

    for filename in options.filenames:

        print(filename)

        rgb = cv2.imread(filename)
        
        if rgb is None:
            print('warning: error opening {}, skipping'.format(filename))
            continue

        cursize = (rgb.shape[1], rgb.shape[0])
        
        if imagesize is None:
            imagesize = cursize
        else:
            assert imagesize == cursize

        print('loaded ' + filename + ' of size {}x{}'.format(*imagesize))

        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        
        retval, corners = cv2.findChessboardCorners(gray, patternsize)

        if options.show_detections:
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(display, patternsize, corners, retval)
            cv2.imshow(win, display)
            cv2.imwrite('/home/mendel/AprilTag_test/Calibration_Images/ChessboardWithCorners.jpg', display)
            cv2.waitKey(5000)

        if not retval:
            print('warning: no chessboard found in {}, skipping'.format(filename))
        else:
            ipoints.append( corners )

    flags = (cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 |
             cv2.CALIB_FIX_K2 |
             cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_K4 |
             cv2.CALIB_FIX_K5 |
             cv2.CALIB_FIX_K6)

    opoints = [opoints] * len(ipoints)

    retval, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(opoints, ipoints, imagesize,
                                                           cameraMatrix=None,
                                                           distCoeffs=np.zeros(5),
                                                           flags=flags)

    assert( np.all(dcoeffs == 0) )
    
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    params = (fx, fy, cx, cy)

    print()
    print('all units below measured in pixels:')
    print('  fx = {}'.format(K[0,0]))
    print('  fy = {}'.format(K[1,1]))
    print('  cx = {}'.format(K[0,2]))
    print('  cy = {}'.format(K[1,2]))
    print()
    print('pastable into Python:')
    print('  fx, fy, cx, cy = {}'.format(repr(params)))
    print()

if __name__ == '__main__':
    main()