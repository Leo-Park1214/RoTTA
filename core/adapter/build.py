from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .potta_wandb import PoTTA
from .raw import RAW

def build_adapter(cfg) -> type(BaseAdapter):
    if "rotta" in cfg.ADAPTER.NAME:
        return RoTTA

    elif "potta" in cfg.ADAPTER.NAME:
        return PoTTA
    elif "raw" in cfg.ADAPTER.NAME:
        return RAW
        
    else:
        raise NotImplementedError("Implement your own adapter")

