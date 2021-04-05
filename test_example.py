import torch
import torchvision

from training_example import classes
from yorutorch import models, datasets, transforms, devices

if __name__ == '__main__':
    model = models.Model.load_from_checkpoint('checkpoints/efficientnet.ckpt').to(device=devices.cuda_otherwise_cpu)
    model.freeze()

    test_dataset = datasets.ImageCSVTestDataset(
        csv_path='data/test.csv',
        key_col_name='image',
        key_to_path_fn=lambda key: f'data/{key}',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.ToCudaOtherwiseCPU()
        ]),
    )

    image_key, image = test_dataset[0]

    image = image.view((1, *image.shape))
    output = torch.softmax(model(image), dim=1)[0]
    print(classes[output.argmax().item()])
