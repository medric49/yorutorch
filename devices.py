import torch

cuda_otherwise_cpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.device('cuda')
cpu = torch.device('cpu')
