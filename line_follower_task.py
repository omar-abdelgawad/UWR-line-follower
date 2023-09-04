import numpy as np
import cv2
from handle_lines import hough_lines_end_points
from color_correct import correct

"""This is a module NOT a script. DON'T RUN THIS MODULE OR USE IT'S MAIN FUNCTION.
The function to use in this module is called [get_thickness_and_direction].
"""

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
LINE_THICKNESS = 4
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1
# tunable parameters
MARGIN = 120
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


def find_slope(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
    """Calculates Slope between two points"""
    return (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])


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


def combine_lines_into_one(
    end_pts: list[list[tuple[int, int]]]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Combines all given lines into one line using average method.

    Args:
        end_pts(list[list[tuple[int,int]]]): list of lines.

    Retruns:
        tuple[tuple[int,int],tuple[int,int]]: two points that represent the combined line.
    """
    x1, y1 = 0, 0
    x2, y2 = 0, 0
    for lst_two_points in end_pts:
        point1, point2 = lst_two_points
        x1 += point1[0]
        y1 += point1[1]
        x2 += point2[0]
        y2 += point2[1]
    new_point1 = (x1 // len(end_pts)), (y1 // len(end_pts))
    new_point2 = (x2 // len(end_pts)), (y2 // len(end_pts))
    return new_point1, new_point2


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
) -> tuple[int, tuple[int, int], dict[str, bool]] | None:
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
    global MARGIN
    img = apply_filter(img)
    mask = get_red_mask(img)
    cx, cy = get_center_moment(mask)
    lines = get_lines(mask)
    if lines is None:
        return None

    end_points = hough_lines_end_points(lines, img.shape)
    new_point1, new_point2 = combine_lines_into_one(end_points)
    slope = find_slope(new_point1, new_point2)
    angle_in_rad = (np.pi / 2) - np.arctan(slope)
    number_of_pix = np.count_nonzero(mask[len(mask) // 2])
    thickness = int(np.abs(number_of_pix * np.cos(angle_in_rad)))
    angle_in_deg = np.degrees(angle_in_rad)

    dir_dict = {
        "center": (0, 0),
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    ret_dict = {}
    cen_row, cen_col = img.shape[0] // 2, img.shape[1] // 2
    while MARGIN >= cen_row or MARGIN >= cen_col:
        MARGIN = MARGIN // 2
    for [direction, (c1, c2)] in dir_dict.items():
        ret_dict[direction] = bool(mask[cen_row + c1 * MARGIN][cen_col + c2 * MARGIN])

    # Debugging
    print(number_of_pix, angle_in_deg, thickness)
    for [direction, (c1, c2)] in dir_dict.items():
        color = BLUE if ret_dict[direction] else GREEN
        cv2.circle(img, (cen_col + c2 * MARGIN, cen_row + c1 * MARGIN), 5, color, 10)

    cv2.arrowedLine(
        img, (img.shape[1] // 2, img.shape[0]), (cx, cy), GREEN, LINE_THICKNESS
    )
    cv2.line(img, new_point1, new_point2, GREEN, LINE_THICKNESS)
    cv2.line(img, (50, 0), (50, thickness), GREEN, LINE_THICKNESS)
    cv2.putText(
        img,
        f"thickness = {thickness}",
        (50, 20),
        FONT,
        FONT_SCALE,
        GREEN,
        5,
    )
    cv2.imshow("image after", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return thickness, (cx, cy), ret_dict


def main(args: (list[str] | None)) -> None:
    # Just testing and debugging. DON'T RUN THIS MODULE
    img = cv2.imread("test_images/red4.webp")
    print("ret value of function: ", get_thickness_and_direction(img))


if __name__ == "__main__":
    main(None)
