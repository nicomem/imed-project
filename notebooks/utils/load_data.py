import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from pathlib import Path

class SlicesSequence(Sequence):
    '''
    Helper class for slices lazy loading.
    Each index access correspond to a batch of scan slices.
    '''

    @staticmethod
    def create_indexes(dataset_nib: dict, shuffle: bool) -> np.ndarray:
        # Get the number of slices for each scan
        scans_len = [scan.shape[-1] for scan in dataset_nib['wmh']]

        # Create the (i_scan, i_slices) array
        ranges = [
            [i_scan, i_slice]
            for i_scan in range(len(scans_len))
            for i_slice in range(scans_len[i_scan])
        ]
        indexes = np.asarray(ranges, dtype=np.uint16)

        # Shuffle the indexes
        np.random.shuffle(indexes)
        return indexes

    def __init__(self, dataset_nib: dict, target_height: int, target_width: int,
                 batch_size: int, shuffle = True):
        self.dataset_nib = dataset_nib
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = SlicesSequence.create_indexes(dataset_nib, shuffle)
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self) -> int:
        # Number of batches: floor(nb_samples / batch_size)
        return len(self.indexes) // self.batch_size

    def __getitem(self, idx: int, fetch_all = False) -> (np.ndarray, np.ndarray):
        # Do not use '3DT1' since its slices shapes are weird
        dtypes = {
            'T1': np.float32,
            'FLAIR': np.float32,
            'wmh': np.bool
        }
        cols_without_target = ['T1', 'FLAIR']

        if fetch_all:
            # Load all slices
            batch_dic = {
                k: np.asarray([
                    np.moveaxis(np.asarray(
                        self.dataset_nib[k][i_scan].dataobj,
                        dtype=dtype
                    ), -1, 0)
                    for i_scan in range(self.dataset_nib[k].shape[0])
                ], dtype=np.ndarray)
                for k,dtype in dtypes.items()
            }

            # Flatten the arrays
            batch_dic = {
                k: np.asarray([
                    sl
                    for slices in batch_dic[k]
                    for sl in slices
                ], dtype=np.ndarray)
                for k in dtypes.keys()
            }
        else:
            # Check for index out of range
            len_self = len(self)
            if idx >= len_self or idx < -len_self:
                raise IndexError('Sequence index out of range')

            # Allow negative index like python list
            if idx < 0:
                idx += len_self

            # Get the current batch
            indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

            # Load all slices for the wanted element
            batch_dic = {
                k: np.asarray([
                    # Load the wanted slice from index=[i_scan, i_slice]
                    np.asarray(
                        self.dataset_nib[k][index[0]].dataobj[...,index[1]],
                        dtype=np.float32
                    )
                    for index in indexes
                ], dtype=np.ndarray)
                for k in dtypes.keys()
            }

        # Reshape the slices
        for k,v in batch_dic.items():
            for i, img in enumerate(v):
                batch_dic[k][i] = tf.image.resize_with_crop_or_pad(
                    img[...,None].astype(dtypes[k]),
                    self.target_height,
                    self.target_width
                )[...,0]

            batch_dic[k] = np.stack(batch_dic[k], axis=0).astype(dtypes[k])

        # Regroup all inputs together
        inputs = np.stack([
            batch_dic[cat]
            for cat in cols_without_target
        ], axis=-1)

        # Add the channels axis (single channel)
        outputs = batch_dic['wmh']

        return inputs, outputs

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        return self.__getitem(idx)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_all(self):
        old_batch_size = self.batch_size

        # Change batch size to get all slices at once
        self.batch_size = len(self.indexes)
        X, Y = self.__getitem(0, fetch_all=True)

        # Restore old batch size
        self.batch_size = old_batch_size

        return X, Y

class CachedSlicesSequence(Sequence):
    def __init__(self, slices_seq: SlicesSequence, batch_size: int, shuffle = True, preprocess = False):
        self.X, self.Y = slices_seq.load_all()
        if preprocess:
            self.X = preprocess_slices(self.X)

        self.indexes = np.arange(0, self.Y.shape[0])
        self.batch_size = batch_size
        self.shuffle = shuffle

        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return self.Y.shape[0] // self.batch_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        # Check for index out of range
        len_self = len(self)
        if idx >= len_self or idx < -len_self:
            raise IndexError('Sequence index out of range')

        # Allow negative index like python list
        if idx < 0:
            idx += len_self

        batch_x = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.Y[idx*self.batch_size:(idx+1)*self.batch_size]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

import cv2
from skimage import morphology
from skimage.morphology import square, disk

def preprocess_slices(slices, disk_kernel = 1, channel = 1):
    """
    Preprocess the dataset by adding a tophat layer

    Parameters:
    -----------
    slices: X
    channel: FLAIR channel
    """
    tophat_kernel = disk(disk_kernel)
    new_shape = (slices.shape[0], slices.shape[1], slices.shape[2], slices.shape[3] + 1)
    preprocessed = np.zeros(new_shape)
    preprocessed[:,:,:,:2] = slices

    for i, im in enumerate(slices):
        tophat_img = morphology.white_tophat(im[:,:,channel], selem=tophat_kernel)

        preprocessed[i,:,:,slices.shape[3]] = tophat_img 
    return preprocessed

def get_dataset(data_dir: str, val_ratio = 0.1, test_ratio = 0.1, verbose = False) -> (dict, dict, dict):
    """
    Load the dataset and split it into train/val/test sets.

    Parameters:
    -----------
    data_dir:
        The path to the data directory.

    val_ratio:
        The proportion of validation data.

    test_ratio:
        The proportion of test data.

    verbose:
        Whether debug information should be printed.

    Returns:
    --------
    train_nib: np.array(nib_image)
        An array of training data files loaded with `nibabel.load`.

    val_nib: np.array(nib_image)
        An array of validation data files loaded with `nibabel.load`.

    test_nib: np.array(nib_image)
        An array of test data files loaded with `nibabel.load`.
    """

    data_dir = Path(data_dir)
    categories = ['3DT1', 'FLAIR', 'T1', 'wmh']

    # Scan the directories for each type of input file
    input_files = {
        cat: list((data_dir / cat).glob('*.nii.gz'))
        for cat in categories
    }

    nb_files_unique = np.unique([len(v) for v in input_files.values()])
    if len(nb_files_unique) != 1:
        raise RuntimeError('Different number of files in each category')

    nb_files = nb_files_unique[0]
    if verbose:
        print('Number of files for each category:', nb_files)

    # Load the files via nibabel
    inputs_nib = {
        k: np.asarray([nib.load(f) for f in v], dtype=np.object)
        for k,v in input_files.items()
    }

    # Split into train/val/test
    shuffle = np.arange(nb_files)
    np.random.shuffle(shuffle)

    nb_val = int(np.ceil(val_ratio * nb_files))
    nb_test = int(np.ceil(test_ratio * nb_files))
    nb_train = nb_files - nb_val - nb_test

    train_index = shuffle[:nb_train]
    val_index = shuffle[nb_train:nb_train+nb_val]
    test_index = shuffle[nb_train+nb_val:]

    if verbose:
        print('Train/val/test split:', nb_train, '/', nb_val, '/', nb_test)

    train_nib = {
        k: v[train_index]
        for k,v in inputs_nib.items()
    }

    val_nib = {
        k: v[val_index]
        for k,v in inputs_nib.items()
    }

    test_nib = {
        k: v[test_index]
        for k,v in inputs_nib.items()
    }

    return (train_nib, val_nib, test_nib)

if __name__ == '__main__':
    train, val, test = get_dataset('../../data', verbose=True)

    # slices = SlicesSequence(train, 200, 200, 3, shuffle=True)
    # slices_cache = CachedSlicesSequence(slices, 3)
    # print(slices_cache.X.shape, slices_cache.X.dtype, slices_cache.X.nbytes / 1_000_000, 'MB')
    # print(slices_cache.Y.shape, slices_cache.Y.dtype, slices_cache.Y.nbytes / 1_000_000, 'MB')
    # x,y = slices_cache[0]
    # print(x.shape)
    # print(x[0].shape)
    # print(y.shape)
    # print(y[0].shape)

