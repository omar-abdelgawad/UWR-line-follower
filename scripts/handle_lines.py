"""
This module contains helper functions to handle geometry related to lines and the Houghlines function in cv2.
"""
import numpy as np


def find_slope(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
    """Calculates Slope between two points"""
    return (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])


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


def polar2cartesian(
    rho: float, theta_rad: float, rotate90: bool = False
) -> tuple[float, float]:
    """
    Converts line equation from polar to cartesian coordinates

    Args:
        rho: input line rho
        theta_rad: input line theta
        rotate90: output line perpendicular to the input line

    Returns:
        m: slope of the line
           For horizontal line: m = 0
           For vertical line: m = np.nan
        b: intercept when x=0
    """
    x = np.cos(theta_rad) * rho
    y = np.sin(theta_rad) * rho
    m = np.nan
    if not np.isclose(x, 0.0):
        m = y / x
    if rotate90:
        if m is np.nan:
            m = 0.0
        elif np.isclose(m, 0.0):
            m = np.nan
        else:
            m = -1.0 / m
    b = 0.0
    if m is not np.nan:
        b = y - m * x

    return m, b


def line_end_points_on_image(
    rho: float, theta: float, image_shape: tuple
) -> list[tuple[float, float]]:
    """
    Returns end points of the line on the end of the image
    Args:
        rho: input line rho
        theta: input line theta
        image_shape: shape of the image

    Returns:
        list: [(x1, y1), (x2, y2)]
    """
    m, b = polar2cartesian(rho, theta, True)
    end_pts = []
    if not np.isclose(m, 0.0):
        x = int(0)
        y = int(solve4y(x, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))
            x = int(image_shape[1] - 1)
            y = int(solve4y(x, m, b))
            if point_on_image(x, y, image_shape):
                end_pts.append((x, y))

    if m is not np.nan:
        y = int(0)
        x = int(solve4x(y, m, b))
        if point_on_image(x, y, image_shape):
            end_pts.append((x, y))
            y = int(image_shape[0] - 1)
            x = int(solve4x(y, m, b))
            if point_on_image(x, y, image_shape):
                end_pts.append((x, y))

    return end_pts


def solve4x(y: float, m: float, b: float) -> float:
    """
    From y = m * x + b
         x = (y - b) / m
    """
    if np.isclose(m, 0.0):
        return 0.0
    if m is np.nan:
        return b
    return (y - b) / m


def solve4y(x: float, m: float, b: float) -> float:
    """
    y = m * x + b
    """
    if m is np.nan:
        return b
    return m * x + b


def point_on_image(x: int, y: int, image_shape: tuple) -> bool:
    """
    Returns true is x and y are on the image
    """
    return 0 <= y < image_shape[0] and 0 <= x < image_shape[1]


def intersection(m1: float, b1: float, m2: float, b2: float) -> tuple[int, int]:
    # Consider y to be equal and solve for x
    # Solve:
    #   m1 * x + b1 = m2 * x + b2
    if np.isclose(m1, m2):
        return (-(10**5), -(10**5))
    x = (b2 - b1) / (m1 - m2)
    # Use the value of x to calculate y
    y = m1 * x + b1

    return int(round(x)), int(round(y))


def hough_lines_end_points(
    lines: np.ndarray, image_shape: tuple
) -> list[list[tuple[int, int]]]:
    """
    Returns end points of the lines on the edge of the image.
    the output list contains a list for every line that contains two points.
    """
    if len(lines.shape) == 3 and lines.shape[1] == 1 and lines.shape[2] == 2:
        lines = np.squeeze(lines, axis=1)
    end_pts = []
    for line in lines:
        rho, theta = line
        new_line = line_end_points_on_image(rho, theta, image_shape)
        if len(new_line) == 2:
            end_pts.append(new_line)
    return end_pts


def hough_lines_intersection(lines: np.ndarray, image_shape: tuple):
    """
    Returns the intersection points that lie on the image
    for all combinations of the lines
    """
    if len(lines.shape) == 3 and lines.shape[1] == 1 and lines.shape[2] == 2:
        lines = np.squeeze(lines, axis=1)
    lines_count = len(lines)
    intersect_pts = []
    for i in range(lines_count - 1):
        for j in range(i + 1, lines_count):
            m1, b1 = polar2cartesian(lines[i][0], lines[i][1], True)
            m2, b2 = polar2cartesian(lines[j][0], lines[j][1], True)
            x, y = intersection(m1, b1, m2, b2)
            if point_on_image(x, y, image_shape):
                intersect_pts.append([x, y])
    return np.array(intersect_pts, dtype=int)
