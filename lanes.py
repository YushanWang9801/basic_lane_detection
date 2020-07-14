import cv2
import numpy as np
import matplotlib.pyplot as plt

# import the test_image
test_image = cv2.imread('test_image.jpg')

# test_image preprocessing
lane_image = np.copy(test_image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
cv2.imwrite("output/gray.jpg", gray)
blur = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imwrite("output/blur.jpg", blur)
canny = cv2.Canny(blur, 50, 150)
cv2.imwrite("output/canny.jpg", canny)

# extract ROI
# manually extract the coordinates of the triangle of the ROI
height = canny.shape[0]
triangle = np.array([[(200,height), (1100,height), (550,250)]])
mask = np.zeros_like(canny)
cv2.fillPoly(mask, triangle, 255)
masked_image = cv2.bitwise_and(canny, mask)
cv2.imwrite("output/masked.jpg", masked_image)

# Hough_transform and display lines
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]),
minLineLength=40, maxLineGap=5)
line_image = np.zeros_like(lane_image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0), 10)
comb_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imwrite("output/result1.jpg", comb_image)

# Optimization
left_fit = []
right_fit = []
for line in lines:
    x1, y1, x2, y2 = line.reshape(4)
    param = np.polyfit((x1,x2), (y1,y2), 1)
    slope = param[0]
    inpt  = param[1]
    if slope < 0:
        left_fit.append((slope, inpt))
    else:
        right_fit.append((slope, inpt))

left_fit_avg = np.average(left_fit, axis = 0)
right_fit_avg = np.average(right_fit, axis = 0)

left_slope, left_inpt = left_fit_avg
left_y1 = lane_image.shape[0]
left_y2 = int(left_y1*(3/5))
left_x1 = int((left_y1-left_inpt)/left_slope)
left_x2 = int((left_y2-left_inpt)/left_slope)
left_line = np.array([left_x1,left_y1,left_x2,left_y2])

right_slope, right_inpt = right_fit_avg
right_y1 = lane_image.shape[0]
right_y2 = int(right_y1*(3/5))
right_x1 = int((right_y1-right_inpt)/right_slope)
right_x2 = int((right_y2-right_inpt)/right_slope)
right_line = np.array([right_x1,right_y1,right_x2,right_y2])

averaged_lines = np.array([left_line, right_line])
line_image2 = np.zeros_like(lane_image)
if averaged_lines is not None:
    for line in averaged_lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image2, (x1,y1),(x2,y2),(255,0,0), 10)
comb_image2 = cv2.addWeighted(lane_image, 0.8, line_image2, 1, 1)
cv2.imwrite("output/opt_result.jpg", comb_image2)

cv2.imshow("result", comb_image2)
cv2.waitKey(0)
