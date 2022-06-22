import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib


def allMatches(kp_left, kp_right, matches):
    # Get all matches coordinates
    all = []
    for i, (m, n) in enumerate(matches):
        all.append(m)

    left_pts = [kp_left[m.queryIdx].pt for m in all]
    right_pts = [kp_right[m.trainIdx].pt for m in all]
    left_array = np.array(left_pts)
    right_array = np.array(right_pts)
    return left_array, right_array


def epipolar(left_array, kp_left, right_array, kp_right, matches, matchesMask):
    # Filter with Epipolar Constraint
    good = []
    for i, (m, n) in enumerate(matches):
        if left_array[i, 1] - right_array[i, 1] == 0:
            matchesMask[i] = [1, 0]
            good.append(m)

    left_pts = [kp_left[m.queryIdx].pt for m in good]
    right_pts = [kp_right[m.trainIdx].pt for m in good]
    left_array = np.array(left_pts)
    right_array = np.array(right_pts)
    return left_array, right_array, matchesMask


def disparityCalc(left_array, right_array):
    # To calculate disparity, first obtain x-y coordinates
    x_L = left_array[:, 0]
    x_R = right_array[:, 0]
    disparity = x_L - x_R
    disp = []
    left = []
    # Filter zero value disparity
    for i in range(len(disparity)):
        if disparity[i] != 0:
            disp.append(disparity[i])
            left.append(left_array[i])

    left = np.array(left)
    disp = np.array(disp)
    return left, disp


def gtData(gt_image_path, left, depth_values):
    # To compare, load ground truth depth map and get real depth values:
    gt_map = cv.imread(gt_image_path, cv.IMREAD_GRAYSCALE).T

    y = left[:, 1]
    x = left[:, 0]
    gt_depths = []
    for i in range(len(depth_values)):
        xx = int(x[i])
        yy = int(y[i])
        gt_depths.append(gt_map[xx, yy])

    gt_depths = np.array(gt_depths)
    return gt_depths


def rmseVals(gt_depths, depth_values):
    prediction = []
    desired = []
    for i in range(len(gt_depths)):
        if gt_depths[i] != 0:
            prediction.append(depth_values[i])
            desired.append(gt_depths[i])

    desired = np.array(desired)
    prediction = np.array(prediction)

    return prediction, desired


def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)

    return rmse_val


# Input
left_image_dir = os.path.abspath('./training/left')
right_image_dir = os.path.abspath('./training/right')
calib_dir = os.path.abspath('./training/calib')
gt_dir = os.path.abspath('./training/gt_depth_map')
#sample_list = ['000010']# Picture by picture analysis
sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']

# ORB and FLANN Parameters
orb = cv.ORB_create(nfeatures=1000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)


# Output
err1 = []
err2 = []
total1 = []
totalEval1 = []
total2 = []
totalEval2 = []

# Main
for sample_name in sample_list:
    # Image and File paths
    left_image_path = left_image_dir + '/' + sample_name + '.png'
    right_image_path = right_image_dir + '/' + sample_name + '.png'
    calib_file_path = calib_dir + '/' + sample_name + '.txt'
    gt_image_path = gt_dir + '/' + sample_name + '.png'

    # Images loading
    img_left = cv.imread(left_image_path, 0)
    img_right = cv.imread(right_image_path, 0)

    # PART 1: FEATURE DETECTION
    kp_left = orb.detect(img_left, None)
    kp_right = orb.detect(img_right, None)
    kp_left, des_left = orb.compute(img_left, kp_left)
    kp_right, des_right = orb.compute(img_right, kp_right)

    # PART 2: FEATURE MATCHING
    matches = flann.knnMatch(des_left, des_right, k=2)

    # Mask Creation for Epipolar Constraint
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Get all matches coordinates
    left_array, right_array = allMatches(kp_left, kp_right, matches)

    # Filter with Epipolar Constraint
    left_array, right_array, matchesMask = epipolar(left_array, kp_left, right_array, kp_right, matches, matchesMask)

    # Read calibration for left camera p2 and right camera p3
    frame_calib = read_frame_calib(calib_file_path)
    stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

    # To calculate disparity, first obtain x-y coordinates
    left, disp = disparityCalc(left_array, right_array)

    # Use Stereo Calibration Parameters to find estimated depth values
    depth_values = (stereo_calib.f * stereo_calib.baseline) / disp

    # To compare, load ground truth depth map and get real depth values:
    gt_depths = gtData(gt_image_path, left, depth_values)

    # Compute RMSE, with consideration of values of 0 depth in GT data
    prediction, desired = rmseVals(gt_depths, depth_values)
    error = rmse(prediction, desired)
    err1.append(error)
    total1.append(len(depth_values))
    totalEval1.append(len(prediction))

    # PART 3: OUTLIER REJECTION
    # Will use function cv2.findHomography to use RANSAC as outlier rejection algorithm
    src_pts, dst_pts = allMatches(kp_left, kp_right, matches)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4)

    # Use the mask to find inlier sets
    in_left = []
    in_right = []
    for i in range(len(mask)):
        if mask[i] == 1:
            in_left.append(src_pts[i])
            in_right.append(dst_pts[i])


    # To calculate disparity, first obtain x-y coordinates
    left_final = np.array(in_left)
    right_final = np.array(in_right)
    left2, disp2 = disparityCalc(left_final, right_final)

    # Use Stereo Calibration Parameters to find estimated depth values
    depth_values2 = (stereo_calib.f * stereo_calib.baseline) / disp2
    gt_depths2 = gtData(gt_image_path, left2, depth_values2)

    # Compute RMSE, with consideration of values of 0 depth in GT data
    prediction2, desired2 = rmseVals(gt_depths2, depth_values2)
    error2 = rmse(prediction2, desired2)
    err2.append(error2)
    total2.append(len(depth_values2))
    totalEval2.append(len(prediction2))

print(np.average(np.array(err2)), np.average(np.array(total2)), np.average(np.array(totalEval2)))
print(totalEval2)