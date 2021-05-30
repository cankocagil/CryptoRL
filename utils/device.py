import torch


def get_device_name():
    return torch.cuda.get_device_name(0)

def get_device():
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)