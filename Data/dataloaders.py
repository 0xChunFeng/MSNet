import numpy as np



from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

from Data.dataset import SegDataset


def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))


    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices





def get_dataloaders(dataste, is_ma, resolution, train_input_paths=None, train_target_paths=None, val_input_paths=None, val_target_paths=None, test_input_paths=None, test_target_paths=None, batch_size=1):

    train_dataloader, test_dataloader, val_dataloader = None, None, None

    if is_ma:
        transform_input4train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((resolution, resolution), antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        transform_input4train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((resolution, resolution), antialias=True),
                transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
                ),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

            ]
        )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Resize((resolution, resolution)),
            transforms.Grayscale(),
        ]
    )


    if train_input_paths != None:
        train_dataset = SegDataset(
            input_paths=train_input_paths,
            target_paths=train_target_paths,
            transform_input=transform_input4train,
            transform_target=transform_target,
            hflip=True,
            vflip=True,
            affine=True,
        )
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=True
        )

    if test_input_paths != None:
        test_dataset = SegDataset(
            input_paths=test_input_paths,
            target_paths=test_target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

    if val_input_paths != None:
        val_dataset = SegDataset(
            input_paths=val_input_paths,
            target_paths=val_target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )
        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

    return train_dataloader, test_dataloader, val_dataloader





