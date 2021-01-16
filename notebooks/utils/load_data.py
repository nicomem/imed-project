import nibabel as nib
import numpy as np
from scipy.sparse import bsr_matrix

from pathlib import Path

def _to_sparse_multiple(nib_images: np.ndarray, dtype, verbose = False) -> np.ndarray:
    """
    Load nibabel images and store them in sparse matrices.
    Each nibabel image contains multiple slices,
        all of an image's slices are stored in a new array.

    Parameters:
    -----------
    nib_images: np.ndarray[nib_image]
        A list of nibabel images

    Returns:
    --------
    sparse_matrices: np.ndarray[np.ndarray[scipy.bsr_matrix]]
        A list of sparse matrices
    """

    # Create an array containing all nib_images ndarrays
    res = np.empty(nib_images.shape[0], dtype=np.ndarray)
    for i, nib_obj in enumerate(nib_images):
        if verbose:
            print(f'{i:>3} / {res.shape[0]}')

        nib_data = np.asarray(nib_obj.dataobj)

        # Create an array containing all nib slices
        res[i] = np.empty(nib_data.shape[0], dtype=np.object)

        # Convert and store each slice
        for j in range(nib_data.shape[-1]):
            res[i][j] = bsr_matrix(nib_data[...,j], dtype=dtype)

    return res

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

def load_dataset_nib(dataset_nib, verbose = False):
    """
    Load a collection of nib objects to sparse matrices.
    """

    dtypes = {
        '3DT1': np.float32,
        'FLAIR': np.float32,
        'T1': np.float32,
        'wmh': np.bool
    }

    # Load the data and convert to sparse matrices
    return {
        k: _to_sparse_multiple(v, dtypes[k], verbose)
        for k,v in dataset_nib.items()
    }

if __name__ == '__main__':
    load_dataset('../data', verbose=True)
