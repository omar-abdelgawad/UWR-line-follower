####!/ usr/bin/env python
import math
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist

video = cv2.VideoCapture(0)
line_follower_node = rospy.Publisher("ROV/line_features", Pose2D, queue_size=10)


def callback(twist_msg):
    if twist_msg.linear.x != 0:
        print("here")
        flag, img = video.read()
        if not flag:
            return
        # img = cv2.bilateralFilter(cv2.imread(path), 9,75,75)
        img = cv2.GaussianBlur(img, (15, 15), cv2.BORDER_DEFAULT)
        # Converting to HSV color space for better red-color segmentation
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Lower and upper HSV bounds for red (at both ends of the spectrum)
        upperb1 = np.array([10, 255, 255])
        lowerb1 = np.array([0, 75, 20])

        lowerb2 = np.array([160, 75, 20])
        upperb2 = np.array([180, 255, 255])

        # Masking the path to be followed
        lower_mask: np.ndarray = cv2.inRange(img_hsv, lowerb1, upperb1)
        upper_mask: np.ndarray = cv2.inRange(img_hsv, lowerb2, upperb2)

        mask = lower_mask + upper_mask

        img_edges = cv2.Canny(mask, 50, 200, None, 3)

        lines = cv2.HoughLines(img_edges, 1, np.pi / 180, 150, None, 0, 0)
        # lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 50, None, 50, 10)

        # if it is not none condition is missing
        if lines is None:
            return
        rho = lines[0][0][0]
        theta = lines[0][0][1]

        theta_line = 90 - math.degrees(theta)
        theta_norm_line = math.atan(
            -1 / math.tan(math.radians(theta_line))
        )  # in radians
        thickness = abs(
            np.count_nonzero(mask[len(mask) - 15]) / math.cos(math.pi - theta_norm_line)
        )

        # Distance from center calculations
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = np.array((int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))))
        pt2 = np.array((int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))))
        center = (img.shape[0] / 2, img.shape[1] / 2)
        distanceFromOrigin = np.cross(pt2 - pt1, pt1 - center) / np.linalg.norm(
            pt1 - pt2
        )
        ########

        # publishing message which is Pose2D
        pos = Pose2D()
        pos.x = distanceFromOrigin
        pos.y = thickness
        pos.theta = math.degrees(theta)
        line_follower_node.publish(pos)


def line_follower():
    rospy.init_node("Line_Follower_Node")
    joystick_bool_topic_name = "Activate_Line_Node"
    rospy.Subscriber("cmd_vel", Twist, callback)
    ################rate needs to be adjusted
    # rate = rospy.Rate(10) #need to ask what is the required frequency for sending to control
    rospy.spin()


def main():
    try:
        line_follower()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
