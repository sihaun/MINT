import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class CIFAR10WithText(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataset = CIFAR10(root='./data', train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        text = CIFAR10_LABELS[label]
        return image, text