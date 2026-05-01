from torchvision import datasets, transforms

tfm = transforms.Compose([transforms.ToTensor()])
datasets.MNIST(root="data", train=True, download=True, transform=tfm)
datasets.MNIST(root="data", train=False, download=True, transform=tfm)
print("MNIST downloaded successfully.")