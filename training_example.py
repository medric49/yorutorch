import pytorch_lightning as pl
from torch import nn, optim
import torchvision
import string
from efficientnet_pytorch import EfficientNet
from . import datasets, transforms, data_modules, models, trainings

classes = string.ascii_lowercase + string.digits


class CharToClass:
    def __call__(self, c):
        return classes.index(c)


if __name__ == '__main__':

    # STEP 1 : Definition of the training data

    # Get the training data
    train_dataset = datasets.ImageCSVTrainDataset(
        csv_path='data/train.csv',
        key_col_name='image',
        targets_col_names=['char'],
        key_to_path_fn=lambda key: f'data/images/{key}',
        target_transform=torchvision.transforms.Compose([
            CharToClass(),
            transforms.ToLong(),
        ])
    )

    # Split the data into the training data and the validation data
    train_dataset, valid_dataset = train_dataset.split(0.2)

    # Affect the transformations (you can also define transformations when you create the dataset)
    train_dataset.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomErasing(p=0.1, scale=(0.02, 0.3)),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((-45, 45))], p=0.3),
            torchvision.transforms.RandomErasing(p=0.1, scale=(0.02, 0.3)),
            torchvision.transforms.RandomResizedCrop(112, scale=(0.6, 1.)),
            torchvision.transforms.RandomErasing(p=0.1, scale=(0.02, 0.3)),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomGrayscale(p=0.1),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.3),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    valid_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create the data module used by the training procedure
    data_module = data_modules.DataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=64,
        num_workers=16,
        valid_num_workers=16)

    # STEP 2 : Definition of the model

    # Create the model to train based on a neural network
    model = models.Model(EfficientNet.from_pretrained('efficientnet-b0', num_classes=36))

    # If you want to load the model from a checkpoint file uncomment the next line
    # model = models.Model.load_from_checkpoint('efficientnet.ckpt')

    # Init parameter to use during the training to optimize the model
    model.init_training_parameters(
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(params=model.parameters(), lr=0.001)
    )

    # STEP 3 : Definition of the trainer

    # Create a trainer (from pytorch_lighting) which will train the model by using the data module
    trainer = pl.Trainer(gpus=-1, max_epochs=30)

    # STEP 4 : Gathering the data, model and trainer in a training instance

    # Create the training. A training gathers a game (the data module), a player (the model) and a trainer (the trainer)
    training = trainings.Training(
        data_module=data_module,
        model=model,
        trainer=trainer,
        checkpoint_filename='efficientnet',
        checkpoint_dir_path='checkpoints'
    )

    # STEP 5 : Launch the training
    training.run()
    training.show_loss_curves()
