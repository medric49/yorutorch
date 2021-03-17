import pytorch_lightning as pl
from matplotlib import pyplot as plt


class Model(pl.LightningModule):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.save_hyperparameters()

        self.criterion = None
        self.optimizer = None

        self.batch_loss_collector = []
        self.train_losses = []
        self.valid_losses = []

    def init_training_parameters(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return self.optimizer

    def on_train_epoch_start(self) -> None:
        self.batch_loss_collector = []

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)

        self.batch_loss_collector.append(loss.item())
        return loss

    def on_train_epoch_end(self, outputs) -> None:
        self.train_losses.append(sum(self.batch_loss_collector)/len(self.batch_loss_collector))

    def on_validation_epoch_start(self) -> None:
        self.batch_loss_collector = []

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.batch_loss_collector.append(loss.item())
        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.valid_losses.append(sum(self.batch_loss_collector)/len(self.batch_loss_collector))

    def plot_losses(self):
        plt.figure()
        plt.plot(range(len(self.train_losses)), self.train_losses, color='red', label='Training error')
        plt.plot(range(len(self.valid_losses)), self.valid_losses, color='blue', label='Validation error')
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
        plt.ylim(0)
        plt.legend()
        plt.show()
