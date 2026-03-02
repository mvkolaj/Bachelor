import numpy as np
import scipy.sparse as sparse
import pickle
from collections import defaultdict

my_graph = np.load("my_graph.npy")

def get_pickle_dataset(filename):
    return pickle.load(open(filename, 'rb'))

datasets = {
    "my_graph": my_graph,
}

print(' '.join(datasets.keys()))

def get_dataset(dataset_name_or_filename):
    print("using {} dataset".format(dataset_name_or_filename))
    if dataset_name_or_filename in datasets.keys():
        print("using builtin dataset")
        return datasets[dataset_name_or_filename]
    else:
        return get_pickle_dataset(dataset_name_or_filename)

