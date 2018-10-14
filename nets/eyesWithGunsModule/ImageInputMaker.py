from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
from GlobalVarsAndLibs import *



# reshape dataset:
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
    return dataset, labels
    
def getNetDataInput():

    pickle_file = './FERDataset/FER.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)

        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']

        del save

        train_dataset, train_labels = reformat(train_dataset, train_labels)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        test_dataset, test_labels = reformat(test_dataset, test_labels)
        print('[imageInputMaker]','Training set', train_dataset.shape, train_labels.shape)
        print('[imageInputMaker]','Validation set', valid_dataset.shape, valid_labels.shape)
        print('[imageInputMaker]','Test set', test_dataset.shape, test_labels.shape)
    return [train_dataset, train_labels,
            valid_dataset, valid_labels,
            test_dataset, test_labels]
