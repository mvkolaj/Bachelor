import numpy as np
import scipy.sparse as sparse
import pickle
from collections import defaultdict


enoa_graph   = np.load("enoa_graph.npy")

def get_pickle_dataset(filename):
    return pickle.load(open(filename, 'rb'))

datasets = {
    "enoa_graph":   enoa_graph,
}

print(' '.join(datasets.keys()))

def get_dataset(dataset_name_or_filename):
    print("using {} dataset".format(dataset_name_or_filename))
    if dataset_name_or_filename in datasets.keys():
        print("using builtin dataset")
        return datasets[dataset_name_or_filename]
    else:
        return get_pickle_dataset(dataset_name_or_filename)