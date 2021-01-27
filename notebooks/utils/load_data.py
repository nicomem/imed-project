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
            np.asarray(self.dataset_nib[col][index[0]].dataobj[...,index[1]], np.float32)[...,None]
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
                scan_slices = (np.asarray(scan.dataobj, dtype=np.float32) for scan in self.dataset_nib[col])
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

            # Normalize the slice (except for wmh)
            if col != 'wmh':
                normalize_img = lambda img: img / np.max(img)
                normalize_imgs = lambda imgs: np.stack([
                    normalize_img(img)
                    for img in imgs
                ])

                scan_slices = map(normalize_imgs, scan_slices)

            # We must end the iterator here, or else python does some weird things
            # such as changing the dtype to np.bool
            scan_slices_dic[col] = list(scan_slices)

        # Special case for 3D slices
        if self.slices3D_radius != 0:
            assert fetch_all, "Only implemented for fetch_all=True"

            # Combine the T1 & FLAIR scan by scan
            scan_X = [
                np.stack([T1, FLAIR], -1).astype(np.float32)
                for T1, FLAIR in zip(scan_slices_dic['T1'], scan_slices_dic['FLAIR'])
            ]
            scan_Y = list(scan_slices_dic['wmh'])

            return scan_X, scan_Y

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
    def __init__(self,
                 slices_seq: SlicesSequence,
                 preprocess = False,
                 remove_no_wmh = False,
                 disk_kernel = 1):
        self.X, self.Y = slices_seq.load_all()

        self.batch_size = slices_seq.batch_size
        self.shuffle = slices_seq.shuffle
        self.slices3D_radius = slices_seq.slices3D_radius

        if self.slices3D_radius == 0:
            self.indexes = np.arange(0, len(self.Y))
            if remove_no_wmh:
                self.__remove_no_wmh_indexes2D()
            if preprocess:
                self.__preprocess_slices2D(disk_kernel=disk_kernel)
        else:
            self.indexes = slices_seq.indexes
            if remove_no_wmh:
                self.__remove_no_wmh_indexes3D()
            if preprocess:
                self.__preprocess_slices3D(disk_kernel=disk_kernel)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return len(self.indexes) // self.batch_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        if self.slices3D_radius == 0:
            return self.__getitem2D(idx)
        else:
            return self.__getitem3D(idx)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __remove_no_wmh_indexes2D(self):
        ''' Remove slices where the wmh is all 0 '''

        with_wmh_mask = np.any(self.Y, axis=(1,2))
        self.X = self.X[with_wmh_mask]
        self.Y = self.Y[with_wmh_mask]
        self.indexes = np.arange(0, len(self.Y))

    def __remove_no_wmh_indexes3D(self):
        ''' Remove slices where the wmh is all 0 '''

        for i_scan in range(len(self.Y)):
            no_wmh_mask = np.logical_not(np.any(self.Y[i_scan], axis=(1,2)))
            no_wmh_idx = np.argwhere(no_wmh_mask)

            self.indexes = np.asarray([
                tup
                for tup in self.indexes
                if not (tup[0] == i_scan and tup[1] in no_wmh_idx)
            ], dtype=np.uint16)

    def __preprocess_slices2D(self, disk_kernel = 1, channel_FLAIR = 1):
        """
        Preprocess the dataset by adding a tophat layer

        Parameters:
        -----------
        disk_kernel: int
            The size of the disk kernel used for the tophat computation.
        channel_FLAIR: int
            The index of the FLAIR channel.
        """

        tophat_kernel = disk(disk_kernel)

        prepro = np.asarray([
            morphology.white_tophat(x_slice[...,channel_FLAIR], selem=tophat_kernel)
            for x_slice in self.X
        ], dtype=self.X.dtype)

        self.X = np.concatenate([self.X, prepro[...,None]], axis=-1)

    def __preprocess_slices3D(self, disk_kernel = 1, channel_FLAIR = 1):
        '''
        Compute and append in-place a preprocessing channel to the inputs.
        Preprocess the dataset by adding a tophat layer.

        Parameters:
        -----------
        disk_kernel: int
            The size of the disk kernel used for the tophat computation.
        channel_FLAIR: int
            The index of the FLAIR channel.
        '''

        # Compute the tophat kernel for the preprocess
        tophat_kernel = disk(disk_kernel)

        # Compute and add as a channel the preprocess image
        for i in range(len(self.X)):
            # Compute all preprocess images for the current scan
            # prepro shape = (S,H,W)
            prepro = np.asarray([
                # Compute the preprocess image for each slice
                morphology.white_tophat(x_slice[...,channel_FLAIR], selem=tophat_kernel)
                for x_slice in self.X[i]
            ], dtype=np.float32)

            # Concatenate the preprocess images to the inputs in the channels axis:
            # X[i] shape   = (S,H,W,nb_channels)
            # prepro shape = (S,H,W,1)
            # result shape = (S,H,W,nb_channels+1)
            self.X[i] = np.concatenate([self.X[i], prepro[...,None]], axis=-1)

    def __getitem2D(self, idx: int) -> (np.ndarray, np.ndarray):
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

    def __getitem3D(self, idx: int) -> (np.ndarray, np.ndarray):
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
        batch_x = np.zeros((self.batch_size, height, width, nb_channels, 2 * self.slices3D_radius + 1), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, height, width), dtype=np.bool)

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
                    batch_x[i,...,channel,i_window] = self.X[i_scan][j_slice,...,channel]

        batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], -1))
        return batch_x, batch_y


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

    # Enable/Disable 3D and preprocessing
    slices3D_radius = 1
    preprocess = True

    slices_seq_uncached = SlicesSequence(val, 100, 200, slices3D_radius=slices3D_radius)
    slices_seq = CachedSlicesSequence(slices_seq_uncached, remove_no_wmh=True, preprocess=preprocess)

    print('Number of trainable slices:', len(slices_seq.indexes))
    print('Number of batch:', len(slices_seq))
    print('Batch size:', slices_seq.batch_size)
    print('Slices not trained per epoch:', len(slices_seq.indexes) - len(slices_seq) * slices_seq.batch_size)

    x,y = slices_seq[0]
    print(x.dtype, y.dtype)
    print(x.shape, y.shape)
    print(x[...,0].max())
    print(x[...,1].max())
    print(x[...,2].max())

    x,y = slices_seq[-1]
    print(x.dtype, y.dtype)
    print(x.shape, y.shape)
