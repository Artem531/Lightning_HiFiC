import pytorch_lightning as pl
from src.helpers.datasets import get_dataset
from torch.utils.data import DataLoader

def get_dataloaders(dataset, mode='train', root=None, shuffle=True, pin_memory=True,
                    batch_size=8, normalize=False, **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"openimages", "jetimages", "evaluation"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    Dataset = get_dataset(dataset)

    if root is None:
        dataset = Dataset(mode=mode, normalize=normalize, **kwargs)
    else:
        dataset = Dataset(root=root, mode=mode, normalize=normalize, **kwargs)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=4,
                      pin_memory=True,
                      )


class LightningOpenImages(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        train_loader = get_dataloaders(self.args.dataset,
                                                root=self.args.dataset_path,
                                                batch_size=self.args.batch_size,
                                                mode='train',
                                                shuffle=True,
                                                normalize=self.args.normalize_input_image)
        return train_loader

    def val_dataloader(self):
        test_loader = get_dataloaders(self.args.dataset,
                                               root=self.args.dataset_path,
                                               batch_size=self.args.batch_size,
                                               mode='validation',
                                               normalize=self.args.normalize_input_image)

        train_loader = get_dataloaders(self.args.dataset,
                                                root=self.args.dataset_path,
                                                batch_size=self.args.batch_size,
                                                mode='train',
                                                normalize=self.args.normalize_input_image)
        return [train_loader, test_loader]

    def test_dataloader(self):
        test_loader = get_dataloaders(self.args.dataset,
                                               root=self.args.dataset_path,
                                               batch_size=self.args.batch_size,
                                               mode='validation',
                                               shuffle=True,
                                               normalize=self.args.normalize_input_image)

        return test_loader
