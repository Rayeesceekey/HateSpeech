import logging
import os

'''Config log file'''
LOG_DIR = "Logs"
def log():
    # log configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)   # generic log level

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
                                "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(LOG_DIR,'object_detection.log'))
    file_handler.setFormatter(formatter)

    file_handler.setLevel(logging.INFO) # log file level
    logger.addHandler(file_handler)

    return logger
_in_logger = log()