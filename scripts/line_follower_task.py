"""This is a module NOT a script. DON'T RUN THIS MODULE OR USE IT'S MAIN FUNCTION.
The function to use in this module is called [get_thickness_and_direction].
"""
import numpy as np
import cv2
from color_correct import correct
from typing import Optional


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RADIUS = 3
LINE_THICKNESS = 4
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1
# tunable parameters
BLUR_KERNEL = (15, 15)
LOWER_BOUND_RED_1 = np.array([0, 75, 20])
UPPER_BOUND_RED_1 = np.array([10, 255, 255])
LOWER_BOUND_RED_2 = np.array([160, 75, 20])
UPPER_BOUND_RED_2 = np.array([180, 255, 255])
LINE_THRESHOLD = 90


def correct_color_underwater(img: np.ndarray) -> np.ndarray:
    """This function shifts the color of the input image to correct the color shift underwater.

    Args:
        img (np.ndarray): input image is in BGR format of any size.

    Returns:
        np.ndarray: color-corrected image of the underwater image.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    corrected_img = correct(img_rgb)
    return corrected_img


def apply_filter(img: np.ndarray) -> np.ndarray:
    """Applies Preprocessing filters such as gaussian blur.

    Args:
        img(np.ndarray): input image to be preprocessed.

    Returns:
        np.ndarray: processed image.
    """
    # TODO: add color correction
    img = correct_color_underwater(img)
    img = cv2.GaussianBlur(img, BLUR_KERNEL, cv2.BORDER_DEFAULT)
    return img


def get_red_mask(img: np.ndarray) -> np.ndarray:
    """Converts image to HSV to extract red color and returns red mask"""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_mask: np.ndarray = cv2.inRange(img_hsv, LOWER_BOUND_RED_1, UPPER_BOUND_RED_1)
    upper_mask: np.ndarray = cv2.inRange(img_hsv, LOWER_BOUND_RED_2, UPPER_BOUND_RED_2)
    mask = lower_mask + upper_mask
    return mask


def get_lines(mask: np.ndarray) -> np.ndarray:
    """Finds lines in the image mask."""
    canny_image = cv2.Canny(
        mask, threshold1=50, threshold2=200, edges=None, apertureSize=3
    )
    lines = cv2.HoughLines(canny_image, 1, np.pi / 180, LINE_THRESHOLD, None, 0, 0)
    return lines


def get_center_moment(mask: np.ndarray) -> tuple[int, int]:
    """Takes as input binary image and returns the center of white pixels.

    Args:
        mask(np.ndarray): Input binary image.

    Returns:
        tuple[int,int]: Cx, and Cy of the image.
    """
    moments = cv2.moments(mask)
    if np.isclose(moments["m00"], 0.0):
        return 0, 0

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    return cX, cY


def get_thickness_and_direction(
    img: np.ndarray,
) -> tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """applies preprocessing, red line detection, and finds thickness for line if exists.
    Returns None if no lines exist.

    Args:
        img(np.ndarray): Input image frame from video feed.

    Returns:
        thickness(int): Thickness of the red line in pixels.
        center_of_line(tuple[int,int]): the x,y coordinate of the center of line.
        ret_dict(dict[str,bool]): Contains boolean values of whether points determined by
        MARGIN are on the red line or not.
    """
    img = apply_filter(img)
    mask = get_red_mask(img)
    cx, cy = get_center_moment(mask)
    cx_up, cy_up = get_center_moment(mask[: mask.shape[0] // 2])
    cx_down, cy_down = get_center_moment(mask[mask.shape[0] // 2 :])

    number_of_pix = np.count_nonzero(mask[int(len(mask) * 0.75)])
    thickness = int(np.abs(number_of_pix))
    return thickness, (cx_up, cy_up), (cx, cy), (cx_down, cy_down + mask.shape[0] // 2)


def main(args: (Optional[list[str]])) -> None:
    # Just testing and debugging. DON'T RUN THIS MODULE
    import os

    root = "test_images"
    for filename in os.listdir(root):
        print(filename)
        path = os.path.join(root, filename)
        img = cv2.imread(path)
        thickness, (nextpt), (middle_pt), (prev_pt) = get_thickness_and_direction(img)

        cv2.circle(img, prev_pt, RADIUS, GREEN, 5)
        cv2.circle(img, middle_pt, RADIUS, GREEN, 5)
        cv2.circle(img, nextpt, RADIUS, GREEN, 5)
        cv2.arrowedLine(img, middle_pt, nextpt, BLUE, 5)
        cv2.putText(
            img,
            f"thick = {thickness}",
            (50, 20),
            FONT,
            FONT_SCALE,
            GREEN,
            5,
        )
        cv2.imshow("image after", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(None)
