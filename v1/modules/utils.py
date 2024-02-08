import numpy as np
import cv2


def get_most_freq_angle(contours, area_thresh):
    """
    Calculates the minimum bounding rectangle for that contour (which gives us its angle) and adds 90
    degrees if its width is smaller than its height (since we want to rotate clockwise). Returns the most
    frequent angle in the list of angles.

    Args:
        contours: Pass in the contours from the image
        area_thresh: Determine if the contour is large enough to be considered

    Returns:
        The most frequent angle in the list of angles
    """
    angles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        width, height = rect[1]
        if width < height:
            angle += 90
        angles.append(int(angle))
    return max(set(angles), key=angles.count)


def rotate_img_without_crop(img, angle):
    """
    The rotate_img_without_crop function takes an image and a rotation angle as input.
    It then rotates the image by the given angle, without cropping any part of it.
    The function returns the rotated image.

    Args:
        img (np.array): Pass in the image that is to be rotated
        angle (float): Rotate the image by a certain angle

    Returns:
        A rotated image without cropping it
    """

    img_height, img_width = img.shape[0], img.shape[1]
    center_y, center_x = img_height//2, img_width//2

    rotation_matrix = cv2.getRotationMatrix2D((center_y, center_x), angle, 1.0)
    cos_rotation = np.abs(rotation_matrix[0][0])
    sin_rotation = np.abs(rotation_matrix[0][1])

    new_img_height = int((img_height * sin_rotation) + (img_width * cos_rotation))
    new_img_width = int((img_height * cos_rotation) + (img_width * sin_rotation))

    rotation_matrix[0][2] += (new_img_width/2) - center_x
    rotation_matrix[1][2] += (new_img_height/2) - center_y

    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_img_width, new_img_height))
    return rotated_img


def rotate_img_crop(img, angle, orig_img_width, orig_img_height):
    """
    The rotate_img_crop function takes in an image, a rotation angle, and the original width and height of the image.
    It then rotates the image by that angle using OpenCV's rotate function. It then crops out any excess pixels from
    the rotated image to return an output with dimensions equal to those of the original input.

    Args:
        img (np.array): Pass in the image that you want to rotate
        angle (float): Rotate the image by a certain angle
        orig_img_width (int): Crop the image to the original width of the image
        orig_img_height (int): Crop the image to the original height

    Returns:
        A rotated image with the same size as the original image
    """

    rotated_img = rotate_img_without_crop(img, angle)
    (height, width) = rotated_img.shape[:2]
    center = (width // 2, height // 2)
    rotated_img = rotated_img[center[1] - orig_img_height // 2: center[1] + orig_img_height//2,
                              center[0] - orig_img_width // 2: center[0] + orig_img_width // 2]
    return rotated_img


def calc_angle_rotation_vertical(angle):
    """
    The calc_angle_rotation_vertical function takes in an angle and returns the angle that is 90 degrees
    counterclockwise from it. This function is used to rotate a rectangle so that its longest side is vertical.

    Args:
        angle: Determine the angle of rotation

    Returns:
        The angle of rotation for the vertical axis
    """

    if -180 < angle < -90 or 90 < angle < 180:
        angle += 90
    elif -90 < angle < 0 or 180 < angle < 270:
        angle -= 90
    elif 0 < angle < 90 or 270 < angle < 360:
        angle -= 90
    elif angle in [90, -90, 180, -180]:
        angle = 0
    return angle
