import os
import torch
from torch.utils.data import Dataset, DataLoader
try:
    import torchvision
    import torchvision.transforms as T
except Exception as e:
    raise ImportError('torchvision is required for this project.') from e
from config import Cfg

class NonceWrapper(Dataset):

    def __init__(self, base: Dataset, seed: int=0):
        self.base = base
        g = torch.Generator(device='cpu')
        g.manual_seed(seed + 12345)
        self.nonces = torch.randint(low=0, high=2 ** 63 - 1, size=(len(base),), generator=g, dtype=torch.int64, device='cpu')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return (x, y, self.nonces[idx])

def build_data(cfg: Cfg):
    tf = T.Compose([T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), T.ToTensor()])
    if cfg.DATA_DIR and os.path.isdir(cfg.DATA_DIR):
        base_train = torchvision.datasets.ImageFolder(cfg.DATA_DIR, transform=tf)
        base_test = torchvision.datasets.ImageFolder(cfg.DATA_DIR, transform=tf)
        num_classes = len(base_train.classes)
    else:
        base_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf)
        base_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf)
        num_classes = 10
    train_ds = NonceWrapper(base_train, seed=cfg.SEED)
    test_ds = NonceWrapper(base_test, seed=cfg.SEED + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    return (train_loader, test_loader, num_classes)
