import pickle
import os

def get_file_path():
    return os.path.dirname(os.path.realpath(__file__))

def unpickle_cds():
    """ unpickle and return child-directed speech (CDS) / training corpus """
    return pickle.load(open(get_file_path() + '/CDS.p', "rb"))