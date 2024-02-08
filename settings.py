import logging
import logging.config
import os
import pickle

import yaml
from v1.modules.singleton import Singleton

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def yaml_config(yaml_file):
    with open(yaml_file, "r") as f:
        dict_config = yaml.safe_load(f.read())
        logging.config.dictConfig(dict_config)


class Settings(metaclass=Singleton):
    def set_env(self, env):
        with open(os.path.join(BASE_DIR, "config/environment.yaml")) as f:
            config = yaml.safe_load(f)

        self.__dict__ = config[env]
        self.env = env
        self.version = os.environ.get("TAG", "0.0.0")
        os.environ["app_name"] = self.app_name
        self.set_logger()

    def load_model(self):
        self.model_path = f"{BASE_DIR}/v1/models/model.pkl"
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

    def set_logger(self):
        yaml_config(os.path.join(BASE_DIR, "config/logging-config.yaml"))
        logging.getLogger().setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
        self.logger = logging.getLogger(__name__)
