import cv2
import numpy as np


def convert_scale(img, alpha, beta):
    """
    The convert_scale function takes in an image, alpha and beta values.
    The function then multiplies the image by the alpha value and adds beta to it.
    It then clips any values that are less than 0 or greater than 255 to be 0 or 255 respectively.
    Finally, it returns a new_img which is of type uint8.

    Args:
        img (np.array): Pass the image to be converted
        alpha (float): Control the contrast of the image
        beta (float): Adjust the brightness of an image

    Returns:
        The image with the new values
    """

    new_img = img * alpha + beta
    new_img = np.clip(new_img, 0, 255)
    return new_img.astype(np.uint8)


def automatic_bright_contrast(img, clip_hist_percnt=25):
    """
    The automatic_bright_contrast function takes an image and a percentage value as input.
    The function then calculates the histogram of the image, and clips it by the given percentage value.
    It then finds the minimum gray level in which 25% of pixels are below this gray level,
    and maximum gray level in which 75% of pixels are below this gray level.
    Then it scales pixel values from minimum to maximum range to 0-255 range using alpha and beta values.

    Args:
        img (np.array): Pass the image to be processed
        clip_hist_percnt (int): Clip the histogram

    Returns:
        The new image, the alpha and beta values
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percnt *= (maximum / 100.0)
    clip_hist_percnt /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percnt:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percnt):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    new_img = convert_scale(img, alpha, beta)
    return new_img, alpha, beta
