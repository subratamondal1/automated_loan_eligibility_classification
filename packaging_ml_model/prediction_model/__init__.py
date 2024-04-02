import os

from prediction_model.config import config

with open(os.path.join(config.PACKAGE_ROOT_PATH,"version")) as f:
    __version__:str = f.read().strip()
