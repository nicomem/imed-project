import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence

from pathlib import Path

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
            k: np.asarray(self.dataset_nib[k][idx].dataobj)
            for k in columns
        }

        # Regroup all inputs together
        inputs = np.stack([
            batch_dic['T1'],
            batch_dic['FLAIR']
        ], axis=-1)

        outputs = batch_dic['wmh']

        return inputs, outputs

    def load_all(self) -> (np.ndarray, np.ndarray):
        X = np.empty(shape=len(self), dtype=np.ndarray)
        Y = np.empty(shape=len(self), dtype=np.ndarray)

        for i, (x,y) in enumerate(self):
            X[i] = x
            Y[i] = y

        return (X, Y)


def get_dataset(data_dir: str, train_ratio = 0.9, verbose = False):
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
