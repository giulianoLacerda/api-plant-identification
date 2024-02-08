# create a generic class
import cv2
import numpy as np
from v1.modules.preprocessing import automatic_bright_contrast
from v1.modules.utils import (
    get_most_freq_angle,
    rotate_img_without_crop, calc_angle_rotation_vertical,
    rotate_img_crop
)


class Segmentation:
    def __init__(self):
        pass

    def _bbox_detection(self, contours, area_thresh):
        """
        The _bbox_detection function takes in a list of contours and an area threshold.
        It then iterates through the contours, calculating the area of each one. Return
        the bbox_list with area greater than the area_thresh.

        Args:
            contours (np.array): Pass in the contours of the image
            area_thresh (float): Filter out contours that are too small

        Returns:
            A list of bounding boxes
        """
        bbox_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_thresh:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            bbox_list.append(box.tolist())
        return bbox_list

    def _line_detection(self, contours, area_thresh, img_width):
        """
        This func fits a line using cv2's fitLine function
        and returns two points that represent this line: (img_width-x) * vy/vx + y for righty and -x*vy/vx+y for lefty
        for each contour in the list of contours that are greater than the area_thresh.

        Args:
            contours (np.array): Pass the contours of the image
            area_thresh (float): Filter out contours that are too small
            img_width (int): Set the righty value of the line

        Returns:
            A list of lines, where each line is a list of two points
        """
        line_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_thresh:
                continue
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((img_width-x)*vy/vx)+y)
            line_list.append([(img_width-1, righty), (0, lefty)])
        return line_list

    def _pre_processing(self, img, percent):
        """
        Calls the automatic_bright_contrast function, which returns a corrected image, alpha, and beta values.
        The _pre_processing function then returns the corrected image.

        Args:
            img (np.array): Pass the image to be processed
            percent (int): Adjust the brightness and contrast of the image

        Returns:
            A corrected image
        """
        corrected_img, alpha, beta = automatic_bright_contrast(img, percent)
        return corrected_img

    def _segment(self, channel, area_thresh, **kwargs):
        """
        The _segment function takes a single channel of an image and returns a binary mask.

        Args:
            channel (np.array): Specify which channel to use for segmentation
            area_thresh (float): Filter out contours that are too small
            **kwargs: Pass a variable number of keyword arguments to a function

        Returns:
            A binary image
        """

        block_thresh = kwargs.get("block_size_thresh", 31)
        C_thresh = kwargs.get("C_thresh", 1)
        open_iter = kwargs.get("open_iter", 15)
        close_iter = kwargs.get("close_iter", 60)
        thresh_type = kwargs.get("thresh_type", cv2.THRESH_BINARY)

        # binarization
        bin_img = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, block_thresh, C_thresh)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

        # lines detection
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        most_freq_angle = get_most_freq_angle(contours, area_thresh)
        angle_rotation = calc_angle_rotation_vertical(most_freq_angle)
        rotated_mask = rotate_img_without_crop(mask, angle_rotation)

        kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
        rotated_mask = cv2.morphologyEx(rotated_mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

        # cv2.imwrite("mask.png", rotated_mask)
        rotated_mask = cv2.morphologyEx(rotated_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        rotated_mask = rotate_img_crop(rotated_mask, -angle_rotation, channel.shape[1], channel.shape[0])
        return rotated_mask

    def _segment_label_c(self, img):

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        cb_blur = cv2.GaussianBlur(cb, (5, 5), 0)
        return self._segment(cb_blur, 500, C_thresh=-5, open_iter=60, thresh_type=cv2.THRESH_BINARY_INV)

    def _segment_label_a(self, img):

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)
        return self._segment(cr_blur, 500, C_thresh=2, thresh_type=cv2.THRESH_BINARY_INV)

    def _segment_label_b(self, img):

        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        s_blur = cv2.GaussianBlur(s, (5, 5), 0)
        return self._segment(s_blur, 500)

    def _segment_label_d(self, img):

        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        h_blur = cv2.GaussianBlur(h, (5, 5), 0)
        return self._segment(h_blur, 500)

    def _post_processing(self, binary_img, bbox=False):
        """
        The _post_processing function takes in a binary image and returns a list of bounding boxes or lines.

        Args:
            binary_img (np.array): Find the contours of the image
            bbox (bool): Determine whether the function should return bounding boxes or lines

        Returns:
            A list of bounding boxes if the bbox parameter is true
        """
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if bbox:
            bbox_list = self._bbox_detection(contours, 500)
            return bbox_list
        lines_list = self._line_detection(contours, 500, binary_img.shape[1])
        return lines_list

    def apply_segment(self, img, label, bbox=False):
        """
        The apply_segment function takes in an image and a label, and returns the segmented image.

        Args:
            img (np.array): Pass the image to be segmented into the function
            label (str): Determine which model to use for segmentation
            bbox (bool): Determine if the bounding box should be returned or not

        Returns:
            A tuple of the segmented image and the bounding box or lines
        """
        if label in ["0"]:
            input_img = self._pre_processing(img, 1)
            output_mask = self._segment_label_a(input_img)
            output = self._post_processing(output_mask, bbox=bbox)
        if label in ["1"]:
            input_img = self._pre_processing(img, 1)
            output_mask = self._segment_label_b(input_img)
            output = self._post_processing(output_mask, bbox=bbox)
        if label in ["2", "4"]:
            input_img = self._pre_processing(img, 1)
            output_mask = self._segment_label_c(input_img)
            output = self._post_processing(output_mask, bbox=bbox)
        if label in ["3"]:
            input_img = self._pre_processing(img, 1)
            output_mask = self._segment_label_d(input_img)
            output = self._post_processing(output_mask, bbox=bbox)
        return output
