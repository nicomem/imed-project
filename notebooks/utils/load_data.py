import cv2
import nibabel as nib
import numpy as np
import tensorflow as tf

from skimage import morphology
from skimage.morphology import square, disk
from tensorflow.keras.utils import Sequence

from pathlib import Path

def ndarray_replace(arr: np.ndarray, orig: int, new: int) -> np.ndarray:
    """
    Replace every element with a specified value by another.

    Parameters:
    -----------
    arr:
        The array to modify (in-place).
    orig:
        The value to replace.
    new:
        The value to insert in place of the other.

    Returns:
    --------
    arr:
        The array that have been modified in-place.
    """
    new_arr = arr.copy()
    new_arr[new_arr == orig] = new
    return new_arr


class SlicesSequence(Sequence):
    '''
    Helper class for slices lazy loading.
    Each index access correspond to a batch of scan slices.

    Parameters:
    -----------
    dataset_nib: dict
        The dataset nib dictionary returned by `get_dataset`.
    target_height: int
        The height of the slices after cropping or padding.
    target_width: int
        The width of the slices after cropping or padding.
    slices3D_radius: int
        The radius of the slices window:
        - 0 -> 2D (window = current) -> X = (S, H, W, 2)
        - 1 -> 3D (window = current + 1 before + 1 after) -> X = (S, H, W, 2, 3)
        - ...
        - N -> 3D (window = current + N before + N after) -> X = (S, H, W, 2, 2*N+1)
    batch_size: int
        The number of data slices in a batch.
    shuffle: bool
        Whether to shuffle the data at init and between epochs.
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

    def __init__(self,
                 dataset_nib: dict,
                 target_height: int,
                 target_width: int,
                 slices3D_radius = 0,
                 batch_size = 32,
                 shuffle = True):
        self.dataset_nib = dataset_nib
        self.target_height = target_height
        self.target_width = target_width
        self.slices3D_radius = slices3D_radius
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = SlicesSequence.create_indexes(dataset_nib, shuffle)

    def __len__(self) -> int:
        # Number of batches: floor(nb_samples / batch_size)
        return len(self.indexes) // self.batch_size

    def __fetch_batch_iter(self, col: str, idx: int):
        # Check for index out of range
        len_self = len(self)
        if idx >= len_self or idx < -len_self:
            raise IndexError('Sequence index out of range')

        # Allow negative index like python list
        if idx < 0:
            idx += len_self

        # Get the current batch indexes
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        # Load all slices for the wanted element
        # Here we return an iterator containing `batch_size` np.ndarray[H,W,1] elements
        return (
            # Load the wanted slice from index=[i_scan, i_slice]
            self.dataset_nib[col][index[0]].dataobj[...,index[1]][...,None].astype(np.float32)
            for index in indexes
        )

    def __getitem(self, idx: int, fetch_all = False) -> (np.ndarray, np.ndarray):
        # Do not use '3DT1' since its slices shapes are weird
        dtypes = {
            'T1': np.float32,
            'FLAIR': np.float32,
            'wmh': np.bool
        }
        cols_without_target = ['T1', 'FLAIR']

        preprocess_slices = {
            'T1': lambda x:x,
            'FLAIR': lambda x:x,
            # Set target class 2 to 0
            'wmh': lambda x: ndarray_replace(x, 2, 0)
        }

        scan_slices_dic = {}
        for col, dtype in dtypes.items():
            if fetch_all:
                # Fetch all slices for each scan
                scan_slices = (np.asarray(scan.dataobj) for scan in self.dataset_nib[col])
            else:
                scan_slices = self.__fetch_batch_iter(col, idx)

            # scan_slices: Iterator[np.ndarray[H,W,S]]

            # Move slices axis
            scan_slices = (np.moveaxis(slices, -1, 0) for slices in scan_slices)

            # scan_slices: Iterator[np.ndarray[S,H,W]]

            # Preprocess the slices
            scan_slices = map(preprocess_slices[col], scan_slices)
            # Set the wanted dtype
            scan_slices = (arr.astype(dtype) for arr in scan_slices)

            # scan_slices: Iterator[np.ndarray[S,H,W]]

            # Reshape the slices
            reshape_imgs = lambda imgs: tf.image.resize_with_crop_or_pad(
                imgs[...,None].astype(dtype),
                self.target_height,
                self.target_width
            )[...,0]
            scan_slices = map(reshape_imgs, scan_slices)

            scan_slices_dic[col] = scan_slices

        # Special case for 3D slices
        if self.slices3D_radius != 0:
            assert fetch_all, "Only implemented for fetch_all=True"

            # Combine the T1 & FLAIR scan by scan
            scan_X = (
                np.stack([T1, FLAIR], -1)
                for T1, FLAIR in zip(scan_slices_dic['T1'], scan_slices_dic['FLAIR'])
            )
            scan_Y = scan_slices_dic['wmh']

            return list(scan_X), list(scan_Y)

        # For 2D, merge all slices together
        scan_slices_dic = {
            col: np.concatenate(list(scan_slices), axis=0)
            for col, scan_slices in scan_slices_dic.items()
        }

        # Regroup all inputs together
        inputs = np.stack([
            scan_slices_dic[cat]
            for cat in cols_without_target
        ], axis=-1)

        # Add the channels axis (single channel)
        outputs = scan_slices_dic['wmh']

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
    def __init__(self, slices_seq: SlicesSequence, preprocess = False):
        self.batch_size = slices_seq.batch_size
        self.shuffle = slices_seq.shuffle

        self.X, self.Y = slices_seq.load_all()
        if preprocess:
            self.X = preprocess_slices(self.X)

        self.indexes = np.arange(0, self.Y.shape[0])

        if self.shuffle:
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


class CachedSlices3DSequence(Sequence):
    def __init__(self, slices_seq: SlicesSequence, preprocess = False):
        self.batch_size = slices_seq.batch_size
        self.shuffle = slices_seq.shuffle
        self.indexes = slices_seq.indexes
        self.slices3D_radius = slices_seq.slices3D_radius

        assert (self.slices3D_radius != 0), "The 3D radius must be strictly greater than 0"

        self.X, self.Y = slices_seq.load_all()
        if preprocess:
            preprocess_slices3D(self.X)

        self.nb_slices = self.indexes.shape[0]

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return self.nb_slices // self.batch_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        # Check for index out of range
        len_self = len(self)
        if idx >= len_self or idx < -len_self:
            raise IndexError('Sequence index out of range')

        # Allow negative index like python list
        if idx < 0:
            idx += len_self

        # [batch_size,2] (i_scan, i_slice)
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        # (S, H, W, 2, 3)
        _, height, width, nb_channels = self.X[0].shape
        batch_x = np.zeros((self.batch_size, height, width, nb_channels, 2 * self.slices3D_radius + 1))
        batch_y = np.zeros((self.batch_size, height, width))

        for i in range(len(indexes)):
            # Get current data specific scan slice
            i_scan  = indexes[i][0]
            i_slice = indexes[i][1]

            # Fill current target slice wmh data
            batch_y[i,...] = self.Y[i_scan][i_slice]

            # Fill the 3D X slices (1 current + radius before + radius after)
            for dslice in range(-self.slices3D_radius, self.slices3D_radius + 1):
                j_slice = i_slice + dslice
                nb_slices = self.X[i_scan].shape[0]

                # If window outside the input data, leave empty (filled with 0)
                if j_slice < 0 or j_slice >= nb_slices:
                    continue

                # Fill the current window element with every input channel
                i_window = dslice + self.slices3D_radius
                for channel in range(nb_channels):
                    batch_x[i,...,channel,i_window] = self.X[i_scan][i_slice,...,channel]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def preprocess_slices(slices, disk_kernel = 1, channel = 1):
    """
    Preprocess the dataset by adding a tophat layer

    Parameters:
    -----------
    slices: X
    channel: FLAIR channel
    """

    tophat_kernel = disk(disk_kernel)
    # TODO: Better mem usage by using np.concat at start to do nb_channels + 1
    new_shape = (slices.shape[0], slices.shape[1], slices.shape[2], slices.shape[3] + 1)
    preprocessed = np.zeros(new_shape)
    preprocessed[:,:,:,:2] = slices

    for i, im in enumerate(slices):
        tophat_img = morphology.white_tophat(im[:,:,channel], selem=tophat_kernel)

        preprocessed[i,:,:,slices.shape[3]] = tophat_img

    return preprocessed

def preprocess_slices3D(X: list, disk_kernel = 1, channel_FLAIR = 1):
    """
    Compute and append in-place a preprocessing channel to the inputs.
    Preprocess the dataset by adding a tophat layer.

    Parameters:
    -----------
    X: List[np.ndarray]
        An array of slices for each scan.
    disk_kernel: int
        The size of the disk kernel used for the tophat computation.
    channel_FLAIR: int
        The index of the FLAIR channel.
    """

    # Compute the tophat kernel for the preprocess
    tophat_kernel = disk(disk_kernel)

    # Compute and add as a channel the preprocess image
    for i, arr in enumerate(X):
        # Compute all preprocess images for the current scan
        # prepro shape = (S,H,W)
        prepro = np.asarray([
            # Compute the preprocess image for each slice
            morphology.white_tophat(x_slice[...,channel_FLAIR], selem=tophat_kernel)
            for x_slice in X[i]
        ], dtype=np.float32)

        # Concatenate the preprocess images to the inputs in the channels axis:
        # X[i] shape   = (S,H,W,nb_channels)
        # prepro shape = (S,H,W,1)
        # result shape = (S,H,W,nb_channels+1)
        X[i] = np.concatenate([X[i], prepro[...,None]], axis=-1)


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

    slices_seq = SlicesSequence(train, 100, 200, slices3D_radius=5)
    slices_3d = CachedSlices3DSequence(slices_seq, True)
    x,y = slices_3d[-1]
    print(x.shape, y.shape)
