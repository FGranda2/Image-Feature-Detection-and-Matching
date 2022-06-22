AER 1515 - PERCEPTION FOR ROBOTICS
ASSIGNMENT # 2

NAME: FRANCISCO GRANDA
STUDENT NUMBER: 1006655941


INCLUDED PYTHON FILES:
'training.py' - used for training image sets.
'testing.py'  - used for testing image sets, it includes code to generate the images in the report and the 'P3_result.txt' file.

*Note: In 'testing.py' file, code lines to draw images are commented. They show the message "Uncomment to draw" to indicate the lines that should be uncommented to get images.

REQUIRED DEPENDENCIES:
As provided in the 'starter_code.py', both included files make use of OpenCV, NumPY, and Matplotlib only. Not additional dependencies are required.

IMPLEMENTED FUNCTIONS:
Various Functions were defined in both files to perform certain tasks with the objective of keeping clean and understandable code.
These functions are:

'allMatches(kp_left, kp_right, matches)' - returns two arrays with x-y coordinates for all matches for both right and left images.
'epipolar(left_array, kp_left, right_array, kp_right, matches, matchesMask)' - applies epipolar constraint and returns left and right filtered arrays with an associated mask. 
'disparityCalc(left_array, right_array)' - computes disparity values and filters in case of zero-value disparity. Returns x-y coordinates array for left image and associated disparity.
'gtData(gt_image_path, left, depth_values)' - loads ground truth depth maps and recovers depth values for provided matches coordinate points. Returns array of depths.


Only for 'training.py':

'rmseVals(gt_depths, depth_values)' - filters out values when ground truth depths are not available (zero-value). Returns prediction and target arrays to use for RMSE.
'rmse(predictions, targets)' - computes RMSE error value.