"""
    Module with functions for reading CIRAF data.
"""
import numpy as np

def read_ciraf_file(file_name):
    """
        Read all data from CIRAF-file.
    """
    import cPickle
    file_desc = open(file_name, 'rb')
    result = cPickle.load(file_desc)
    file_desc.close()
    return result

def merge_dicts(*dict_args):
    """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def read_ciraf_10(files_path, count=-1):
    """
        Read CIRAF-10 data in the path.
        Returns list of training batches and test batch.
        Labels in dict are: 'data', 'labels', 'batch_label', 'filenames'
    """
    data_batch_1 = read_ciraf_file(files_path + "/data_batch_1")
    data_batch_2 = read_ciraf_file(files_path + "/data_batch_2")
    data_batch_3 = read_ciraf_file(files_path + "/data_batch_3")
    data_batch_4 = read_ciraf_file(files_path + "/data_batch_4")
    data_batch_5 = read_ciraf_file(files_path + "/data_batch_5")

    test_batch = read_ciraf_file(files_path + "/test_batch")
    train_batches = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]

    if count >= 0 and count < len(train_batches):
        train_batches = train_batches[0: count]

    return train_batches, test_batch

