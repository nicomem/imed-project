import nibabel as nib
import numpy as np
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

    def __init__(self, dataset_nib: dict, batch_size: int, shuffle = True):
        self.dataset_nib = dataset_nib
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.indexes = SlicesSequence.create_indexes(dataset_nib, shuffle)

    def __len__(self):
        # Number of batches: floor(nb_samples / batch_size)
        return len(self.indexes) // self.batch_size

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        # Check for index out of range
        len_self = len(self)
        if idx >= len_self or idx < -len_self:
            raise IndexError('Sequence index out of range')

        # Allow negative index like python list
        if idx < 0:
            idx += len_self

        # Get the current batch
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        # Do not use '3DT1' since its slices shapes are weird
        columns = ['T1', 'FLAIR', 'wmh']

        # Load all slices for the wanted element
        batch_dic = {
            k: np.asarray([
                # Load the wanted slice from index=[i_scan, i_slice]
                np.asarray(self.dataset_nib[k][index[0]].dataobj[...,index[1]])
                for index in indexes
            ], dtype=np.ndarray)
            for k in columns
        }

        # TODO: Reshape the slices

        # Regroup all inputs together
        inputs = np.stack([
            batch_dic['T1'],
            batch_dic['FLAIR']
        ], axis=-1)

        # Add the channels axis (single channel)
        outputs = batch_dic['wmh']

        return inputs, outputs

    def on_epoch_end(self):
        # Shuffle the indexes
        np.random.shuffle(self.indexes)


class NibDataSequence(Sequence):
    '''
    Helper class for dataset lazy loading.
    Each index access correspond to a load of a scan (multiple slices).
    '''

    def __init__(self, dataset_nib):
        self.dataset_nib = dataset_nib
        self.len = len(dataset_nib['T1'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Do not use '3DT1' since its slices shapes are weird
        columns = ['T1', 'FLAIR', 'wmh']

        # Load all slices for the wanted element
        batch_dic = {
            # Change from (Y,X,S) to (S,Y,X) for easier usage
            k: np.moveaxis(
                # Load the slices
                np.asarray(self.dataset_nib[k][idx].dataobj),
                -1,
                0
            )
            for k in columns
        }

        # Regroup all inputs together
        inputs = np.stack([
            batch_dic['T1'],
            batch_dic['FLAIR']
        ], axis=-1)

        # Add the channels axis (single channel)
        outputs = batch_dic['wmh'][...,None]

        return inputs, outputs

    def load_all(self) -> (np.ndarray, np.ndarray):
        X = np.empty(shape=len(self), dtype=np.ndarray)
        Y = np.empty(shape=len(self), dtype=np.ndarray)

        for i, (x,y) in enumerate(self):
            X[i] = x
            Y[i] = y

        return (X, Y)


class CachedDataSequence(Sequence):
    def __init__(self, nib_seq: NibDataSequence):
        self.X, self.Y = nib_seq.load_all()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_dataset(data_dir: str, train_ratio = 0.9, verbose = False) -> (dict, dict):
    """
    Load the dataset and split it into train/validation sets.

    Parameters:
    -----------
    data_dir:
        The path to the data directory.

    train_ratio:
        The proportion of train data (compared to validation).

    verbose:
        Whether debug information should be printed.

    Returns:
    --------
    train_nib: np.array(nib_image)
        An array of training data files loaded with `nibabel.load`.

    val_nib: np.array(nib_image)
        An array of validation data files loaded with `nibabel.load`.
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

    # Split into train/val
    shuffle = np.arange(nb_files)
    np.random.shuffle(shuffle)

    nb_train = int(train_ratio * nb_files)
    train_index = shuffle[:nb_train]
    val_index = shuffle[nb_train:]

    if verbose:
        print('Train/val split:', nb_train, '/', nb_files - nb_train)

    train_nib = {
        k: v[train_index]
        for k,v in inputs_nib.items()
    }

    val_nib = {
        k: v[val_index]
        for k,v in inputs_nib.items()
    }

    return (train_nib, val_nib)

from timeit import timeit
if __name__ == '__main__':
    train, val = get_dataset('../../data')

    slices = SlicesSequence(train, 32, shuffle=False)
    print(slices[0])

    print('Ordered')
    slices = SlicesSequence(train, 32, shuffle=False)
    print(timeit(lambda: slices[0], number=5))

    print('----------')

    print('Shuffled')
    slices = SlicesSequence(train, 32, shuffle=True)
    print(timeit(lambda: slices[0], number=5))

    # i = 0
    # while True:
    #     x,y = slices[i % len(slices)]
    #     print(i, x.shape, y.shape)
    #     i += 1
