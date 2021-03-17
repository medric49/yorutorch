class Training:
    def __init__(self,
                 data_module,
                 model,
                 trainer,
                 checkpoint_filename=None,
                 checkpoint_dir_path=None,
                 checkpoint_callbacks=None
                 ):

        self.data_module = data_module
        self.model = model
        self.trainer = trainer

        self.trainer.checkpoint_callback.monitor = 'val_loss'
        self.trainer.checkpoint_callback.dirpath = checkpoint_dir_path
        self.trainer.checkpoint_callback.filename = checkpoint_filename

        if checkpoint_callbacks is not None:
            for callback in self.trainer.checkpoint_callbacks:
                self.trainer.callbacks.remove(callback)
            for callback in checkpoint_callbacks:
                self.trainer.callbacks.append(callback)

    def run(self):
        self.trainer.fit(self.model, self.data_module)

    def show_loss_curves(self):
        self.model.plot_losses()
