import cv2
import numpy as np


def preprocess(image):
    lane_image = np.copy(image)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def roi(canny):
    height = canny.shape[0]
    triangle = np.array([[(200,height), (1100,height), (550,250)]])
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def optimize_line(frame, lines):
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

    if len(left_fit) != 0 :
        left_fit_avg = np.average(left_fit, axis = 0)
    else :
        left_fit_avg = [-1.70816864, 1273.71047431]
    if len(left_fit) != 0 :
        right_fit_avg = np.average(right_fit, axis = 0)
    else :
        right_fit_avg = [1, -300]
    #print(right_fit_avg)
    left_slope = left_fit_avg[0]
    left_inpt = left_fit_avg[1]
    left_y1 = frame.shape[0]
    left_y2 = int(left_y1*(3/5))
    left_x1 = int((left_y1-left_inpt)/left_slope)
    left_x2 = int((left_y2-left_inpt)/left_slope)
    left_line = np.array([left_x1,left_y1,left_x2,left_y2])

    right_slope = right_fit_avg[0]
    right_inpt = right_fit_avg[1]
    right_y1 = frame.shape[0]
    right_y2 = int(right_y1*(3/5))
    right_x1 = int((right_y1-right_inpt)/right_slope)
    right_x2 = int((right_y2-right_inpt)/right_slope)
    right_line = np.array([right_x1,right_y1,right_x2,right_y2])
    return np.array([left_line, right_line])

# below is to test the algorithm by video
cap = cv2.VideoCapture("test.mp4")
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                        20.0, size)
while(cap.isOpened()):
    ret, frame  = cap.read()
    if ret == True:
        pro_frame = preprocess(frame)
        roi_frame = roi(pro_frame)
        lines = cv2.HoughLinesP(roi_frame, 2, np.pi/180, 100, np.array([]),
        minLineLength=40, maxLineGap=5)
        averaged_lines = optimize_line(frame, lines)
        line_image = np.zeros_like(frame)
        if averaged_lines is not None:
            for line in averaged_lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0), 10)
        comb_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        out.write(comb_frame)
        cv2.imshow("result", comb_frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()