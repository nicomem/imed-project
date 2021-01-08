import nibabel as nib
import numpy as np

from pathlib import Path

def load_dataset(data_dir: str, train_ratio = 0.9, verbose = False):
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
        k: [nib.load(f) for f in v]
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
        k: np.asarray(v)[train_index]
        for k,v in inputs_nib.items()
    }

    val_nib = {
        k: np.asarray(v)[val_index]
        for k,v in inputs_nib.items()
    }

    return (train_nib, val_nib)
