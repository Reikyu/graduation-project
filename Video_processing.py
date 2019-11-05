# import the library
import pyrealsense2 as rs
import numpy as np
import cv2
#import serial

#Create serial
#ser = serial.Serial("/dev/", 115200)
#ser.open()

#setting the roi for camshift
img = cv2.imread("taget.jpg")
roi = img[330: 680, 185: 535]
x = 185
y = 330
width = 535 - x
height = 680 - y
trackWindow = (x, y, width, height)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 30
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# When you left click on the mouse, you save the data to the variable below
# If you want to pre-specify colors, you can change this variable to a value that you clicked with the mouse
hsv = 20
lower_select1 = (20, 150, 150)
upper_select1 = (30, 255, 255)
lower_select2 = (10, 150, 150)
upper_select2 = (20, 255, 255)
lower_select3 = (10, 150, 150)
upper_select3 = (20, 255, 255)

# for creating a trackbar
def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_select1, upper_select1, lower_select2, upper_select2, lower_select3, upper_select3, threshold

    # Left-click on the mouse to read and convert the pixel value in the position to HSV
    if event == cv2.EVENT_LBUTTONDOWN:
        print(bg_removed[y, x])
        color = bg_removed[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        # Converts the current value of the trackbar as the threshold value
        threshold = cv2.getTrackbarPos('threshold', 'img_result')
        # Set a range of pixel values similar to values obtained by clicking the mouse in the HSV color space.
        if hsv[0] < 10:
            print("case1")
            lower_select1 = np.array([hsv[0] - 10 + 180, threshold, threshold])
            upper_select1 = np.array([180, 255, 255])
            lower_select2 = np.array([0, threshold, threshold])
            upper_select2 = np.array([hsv[0], 255, 255])
            lower_select3 = np.array([hsv[0], threshold, threshold])
            upper_select3 = np.array([hsv[0] + 10, 255, 255])

        elif hsv[0] > 170:
            print("case2")
            lower_select1 = np.array([hsv[0], threshold, threshold])
            upper_select1 = np.array([180, 255, 255])
            lower_select2 = np.array([0, threshold, threshold])
            upper_select2 = np.array([hsv[0] + 10 - 180, 255, 255])
            lower_select3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_select3 = np.array([hsv[0], 255, 255])

        else:
            print("case3")
            lower_select1 = np.array([hsv[0], threshold, threshold])
            upper_select1 = np.array([hsv[0] + 10, 255, 255])
            lower_select2 = np.array([hsv[0] - 10, threshold, threshold])
            upper_select2 = np.array([hsv[0], 255, 255])
            lower_select3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_select3 = np.array([hsv[0], 255, 255])

        # Outputs a parameter
        print(hsv)
        print("@1", lower_select1, "~", upper_select1)
        print("@2", lower_select2, "~", upper_select2)
        print("@3", lower_select3, "~", upper_select3)
    return

# Create a trackbar for adjusting the settings and values
cv2.namedWindow('img_result')
cv2.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'img_result', 135)
outb = " "

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 480x270 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 480x270 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((bg_removed, depth_colormap))
        # Converts the original image to HSV images
        img_hsv = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)

        # Generates a mask from the HSV image with the range value.
        img_mask1 = cv2.inRange(img_hsv, lower_select1, upper_select1)
        img_mask2 = cv2.inRange(img_hsv, lower_select2, upper_select2)
        img_mask3 = cv2.inRange(img_hsv, lower_select3, upper_select3)
        img_mask = img_mask1 | img_mask2 | img_mask3

        # Use a morphology operation for noise rejection
        kernel = np.ones((11, 11), np.uint8)
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

        img_mask = cv2.erode(img_mask, None, iterations=1)
        img_mask = cv2.dilate(img_mask, None, iterations=1)

        # Acquire the portion of the image corresponding to the range value from the original image with the mask image.
        img_result = cv2.bitwise_and(color_image, color_image, mask=img_mask)

        # Labeling for tracking object locations
        num0fLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(img_mask)

        for idx, centroids in enumerate(centroids):
            # Parameter setting
            centerX, centerY = int(centroids[0]), int(centroids[1])
            center = (centerX, centerY)
            C = (int(centerX), int(centerY))
            dist = aligned_depth_frame.get_distance(centerX, centerY)
            x, y, width, height, area = stats[idx]
            if stats[idx][0] == 0 and stats[idx][1] == 0:
                if len(stats) == 1:
                    out = "x"
                    if out[0] is not outb[0]:
                        outb = out
                        print(out)
                        #ser.write(out)
                continue

            if np.any(np.isnan(centroids)):
                continue

            dst = cv2.calcBackProject([img_hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(bg_removed, [pts], True, (0, 255, 0), 2)

            if area > 0:
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                ordered_points = np.zeros((4, 2), dtype="float32")
                ordered_points[0] = pts[np.argmin(s)]
                ordered_points[2] = pts[np.argmax(s)]
                ordered_points[1] = pts[np.argmin(diff)]
                ordered_points[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = ordered_points
                max_x = max(tl[0], tr[0], br[0], bl[0])
                min_x = min(tl[0], tr[0], br[0], bl[0])
                max_y = max(tl[1], tr[1], br[1], bl[1])
                min_y = min(tl[1], tr[1], br[1], bl[1])

                if min_x < centerX < max_x and min_y < centerY < max_y:
                    # Displays the outline and center of the object
                    cv2.circle(bg_removed, (centerX, centerY), 2, (0, 0, 255), 3)
                    cv2.circle(bg_removed, (centerX, centerY), int(width / 2), (0, 255, 255), 2)
                    # Displays coordinate values and distances at the center of an object
                    cv2.putText(bg_removed, str(C), (int(centerX), int(centerY)), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 255, 255), 2)
                    cv2.putText(bg_removed, str(dist), (int(centerX), int(centerY) + 30),
                                cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 255, 255), 2)

                    if dist is not 0:
                        if dist < 0.4:
                            out = "b"

                        elif 0.4 < dist < 0.5:
                            out = "s"

                        elif dist > 0.5:
                            if 110 <= centerX <= 310:
                                v = int((dist - 0.5) * 4000)
                                vel = str(int(v / 100) + 1)
                                out = "g" + vel

                            elif centerX < 110:
                                vel = str(int((210 - centerX) * 3 / 100))
                                out = "l" + vel

                            elif centerX > 310:
                                vel = str(int((centerX - 210) * 3 / 100))
                                out = "r" + vel
                    if (out[0] is str("g") or out[0] is str("r") or out[0] is str("l")):
                        if out[0] is not outb[0]:
                            outb = out
                            print(out)
                            # ser.write(out)
                        elif out[0] is outb[0] and out[1] is not outb[1]:
                            outb = out
                            print(out)
                            # ser.write(out)
                    elif out[0] is str("s") or out[0] is str("b"):
                        if out is not outb:
                            outb = out
                            print(out)
                            # ser.write(out)

        #Displays the winodws
        cv2.namedWindow('bg_removed')
        cv2.setMouseCallback('bg_removed', mouse_callback)
        cv2.imshow('bg_removed', bg_removed)
        cv2.imshow('img_result', img_result)

        # Press esc to close the image windows
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
    #ser.close()
