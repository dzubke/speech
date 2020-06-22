# standard libraries
import json


class Config:

    def __init__(self, json_path:str):
        """
        Creates a config object by reading the parameters from the json_path
        """
        with open(json_path, 'r') as fid:
            config = json.load(fid)
        
        self.json_path = json_path
        self.config = config
        self.data_cfg = config["data"]
        self.log_cfg = config["logger"]
        self.preproc_cfg = config["preproc"]
        self.opt_cfg = config["optimizer"]
        self.model_cfg = config["model"]

    def __str__(self):
        return "Incomplete Config __str__ method"
