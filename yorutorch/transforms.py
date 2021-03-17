import torch

from yorutorch import devices


class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, pic):
        return pic.to(device=self.device)


class ToCudaOtherwiseCPU(ToDevice):
    def __init__(self):
        super(ToCudaOtherwiseCPU, self).__init__(device=devices.cuda_otherwise_cpu)


class ToFloat:
    def __call__(self, t):
        return torch.tensor(t, dtype=torch.float)


class ToLong:
    def __call__(self, t):
        return torch.tensor(t, dtype=torch.long)


class ToInt:
    def __call__(self, t):
        return torch.tensor(t, dtype=torch.int)
