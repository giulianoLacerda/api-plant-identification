import logging
import cv2
import numpy as np
from base64 import b64decode
from v1.modules.classifier import Classifier
from v1.modules.segmentation import Segmentation


class PlantSegmentation:
    """This class implements an plant image segmentation routine.

    Attributes:
        logger (logging.Logger): Logger object.
        cfg (dict): A dictionary config.
        classifier (Classifier): Image type classifier model.
    """

    def __init__(self, cfg, model):
        """
        Args:
            logger (logging.Logger): Logger object.
            cfg (dict): A dictionary config.
            classifier (Classifier): Image type classifier model.
        """
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.classifier = Classifier(model)
        self.segmentation = Segmentation()

    def main_routine(self, input):
        """Main routine for plant segmentation.

        Args:
            input (dict): A dictionary with input data

        Returns:
            tuple(pred_label, list): The predicted label and the list of bbox or lines
        """

        if not input:
            self.logger.info("Input is empty")
            return None

        nparr = np.fromstring(b64decode(input["base64"]), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mean_color = np.mean(img_rgb, axis=(0, 1))
        # std_color = np.std(img_rgb, axis=(0, 1))
        # features = np.concatenate([mean_color, std_color])
        features = mean_color
        pred_label = str(self.classifier.predict(list(features)))

        output_list = self.segmentation.apply_segment(img_bgr, pred_label, bbox=input["bbox"])

        # img_bgr_copy = img_bgr.copy()
        # for out in output_list:
        #     cv2.line(img_bgr_copy, out[0], out[1], (0, 0, 255), 1)
        # cv2.imwrite("output.jpg", img_bgr_copy)
        return pred_label, output_list
