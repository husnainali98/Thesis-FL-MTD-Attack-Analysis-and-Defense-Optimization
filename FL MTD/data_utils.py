
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_data_for_client(cid: int, batch_size: int = 32):
    tfm = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    # make number of splits match the actual number of clients
    NUM_CLIENTS = 20  # <-- change this to 6,7,8... if increase in clients

    if cid < 0 or cid >= NUM_CLIENTS:
        raise ValueError(f"cid={cid} out of range. Valid range: 0..{NUM_CLIENTS-1}")

    # 60k train split into NUM_CLIENTS parts (equal sizes)
    base = len(train_ds) // NUM_CLIENTS
    parts = [base] * NUM_CLIENTS
    parts[-1] += len(train_ds) - sum(parts)  # put remainder in last split

    splits = random_split(train_ds, parts, generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(splits[cid], batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
