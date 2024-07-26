# from .KITTILoader import KITTILoader
from .ZEDLoader import KITTILoader

from .SequenceImageLoader import SequenceImageLoader
from .TUMRGBLoader import TUMRGBLoader
import logging, coloredlogs

def create_dataloader(conf):
    logging.warning(f"Inside create_dataloader!")
    try:
        code_line = f"{conf['name']}(conf)"
        logging.info(f"code_line: {code_line}")
        
        loader = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return loader
