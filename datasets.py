import csv

from PIL import Image
from torch.utils.data import dataset
import torch


class AbstractImageCSVDataset(dataset.Dataset):
    def __init__(self, **kwargs):
        super(AbstractImageCSVDataset, self).__init__()
        self.data = kwargs.get('data')
        self.csv_path = kwargs.get('csv_path')
        self.transform = kwargs.get('transform')
        self.key_to_path_fn = kwargs.get('key_to_path_fn')
        self.key_col_name = kwargs.get('key_col_name')

        self.others_col_names = kwargs.get('others_col_names', [])

    def dataloader(self, batch_size=1, shuffle=False, sampler=None, num_workers=0):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)

    def split(self, ratio):
        limit = int(ratio * len(self))
        return self[limit:], self[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError


class ImageCSVTrainDataset(AbstractImageCSVDataset):
    def __init__(self, **kwargs):
        super(ImageCSVTrainDataset, self).__init__(**kwargs)

        self.target_transform = kwargs.get('target_transform')

        if self.data is not None:
            return

        self.data = []
        self.targets_col_names = kwargs.get('targets_col_names')

        csv_file = open(self.csv_path)
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)

        key_pos = header.index(self.key_col_name)
        targets_pos = [header.index(p) for p in self.targets_col_names]

        others_pos = [header.index(p) for p in self.others_col_names]

        for csv_row in csv_reader:
            image_key = csv_row[key_pos]
            targets = [csv_row[p] for p in targets_pos]
            others = [csv_row[p] for p in others_pos]

            if len(targets) == 1:
                targets = targets[0]

            if len(others) == 1:
                others = others[0]
            elif len(others) == 0:
                others = None

            image_path = self.key_to_path_fn(image_key) if self.key_to_path_fn is not None else image_key
            if others is not None:
                self.data.append((image_path, targets, others))
            else:
                self.data.append((image_path, targets))
        csv_file.close()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ImageCSVTrainDataset(data=self.data[item], transform=self.transform, target_transform=self.target_transform)

        image = self.data[item][0]
        target = self.data[item][1]

        image = Image.open(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, *self.data[item][2:]


class ImageCSVTestDataset(AbstractImageCSVDataset):
    def __init__(self, **kwargs):
        super(ImageCSVTestDataset, self).__init__(**kwargs)

        if self.data is not None:
            return

        self.data = []

        csv_file = open(self.csv_path)
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)

        key_pos = header.index(self.key_col_name)

        others_pos = [header.index(p) for p in self.others_col_names]

        for csv_row in csv_reader:
            image_key = csv_row[key_pos]
            others = [csv_row[p] for p in others_pos]

            image_path = self.key_to_path_fn(image_key) if self.key_to_path_fn is not None else image_key

            if len(others) == 1:
                others = others[0]
            elif len(others) == 0:
                others = None

            if others is not None:
                self.data.append((image_key, image_path, others))
            else:
                self.data.append((image_key, image_path))

        csv_file.close()

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ImageCSVTestDataset(data=self.data[item], transform=self.transform)

        image_key = self.data[item][0]
        image = self.data[item][1]

        image = Image.open(image)

        if self.transform is not None:
            image = self.transform(image)

        return image_key, image, *self.data[item][2:]
