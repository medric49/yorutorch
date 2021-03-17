import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, batch_size, shuffle=True, valid_shuffle=False, num_workers=0, valid_num_workers=0):
        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.valid_shuffle = valid_shuffle

        self.num_workers = num_workers
        self.valid_num_workers = valid_num_workers

    def train_dataloader(self):
        return self.train_dataset.dataloader(batch_size=self.batch_size, shuffle=self.shuffle,
                                             num_workers=self.num_workers)

    def val_dataloader(self):
        return self.valid_dataset.dataloader(batch_size=self.batch_size, shuffle=self.valid_shuffle,
                                             num_workers=self.valid_num_workers)
