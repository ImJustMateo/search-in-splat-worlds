
import os
from . import unisal_file_opener
from . import transal_file_opener
from . import tempsal_file_opener



def load_dataloader(args):
    path_ = os.path.dirname(os.path.abspath(__file__))
    print(path_)

    if args.model == "Unisal":
        file_opener = unisal_file_opener
    elif args.model == "TranSalNetDense":
        file_opener = transal_file_opener
    elif args.model == "TranSalNetRes":
        file_opener = transal_file_opener
    elif args.model == "TempSal":
        file_opener = tempsal_file_opener
    return file_opener



def load_dataloader2(model_name):
    path_ = os.path.dirname(os.path.abspath(__file__))
    print(path_)

    if model_name == "Unisal":
        file_opener = unisal_file_opener
    elif model_name == "TranSalNetDense":
        file_opener = transal_file_opener
    elif model_name == "TranSalNetRes":
        file_opener = transal_file_opener
    elif model_name == "TempSal":
        file_opener = tempsal_file_opener
    return file_opener
