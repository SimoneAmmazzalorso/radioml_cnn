import keras
# import os.path

import numpy as np
from PIL import Image

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, path_data, dim=(1000,2000), batch_size=100, n_channels=1, N_out=10, shuffle=True, norm=True):
        'Initialization'
        self.dim = dim
        self.path_data = path_data
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.norm = norm
        self.N_out = N_out
        self.on_epoch_end()

    def __len__(self):
        '''
        Denotes the number of batches per epoch
        A common practice is to set this value to (#samples/batch_size) 
        so that the model sees the training samples at most once per epoch.
        '''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        dtype = np.float32
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=dtype)
        y = np.zeros((self.batch_size, self.N_out), dtype=dtype)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.array(Image.open(self.path_data+ID+'.tif'), dtype=dtype)[...,np.newaxis]

            # Store label
            y[i] = self.labels[ID]
            if self.norm == True:
                X[i,] = X[i,]/255.0     # norming pixel values to 0.0 ... 1.0
                # y[i] = y[i]/np.max(self.labels)
                # Don't do that here, you need to know the maximum value for later purposes!
        return X, y



