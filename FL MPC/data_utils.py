import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_data_for_client(cid: int, batch_size: int = 32):
    tfm = transforms.Compose([transforms.ToTensor()])

    full_train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    NUM_CLIENTS = 10
    
    # Take only first 30,000 images (3,000 per client × 10 clients)
    train_ds, _ = random_split(full_train_ds, [30000, 30000], generator=torch.Generator().manual_seed(42))

    if cid < 0 or cid >= NUM_CLIENTS:
        raise ValueError(f"cid={cid} out of range. Valid range: 0..{NUM_CLIENTS-1}")

    # Split 30,000 images into 10 parts of 3,000 each
    parts = [3000] * NUM_CLIENTS

    splits = random_split(train_ds, parts, generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(splits[cid], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return trainloader, testloader