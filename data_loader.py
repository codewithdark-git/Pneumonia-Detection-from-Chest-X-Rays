from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_dataset(main_dir, batch_size):
    """
    Load datasets organized in 'train', 'val', and 'test' subdirectories.

    Args:
        main_dir (str): Path to the main dataset directory.
        batch_size (int): Batch size for data loading.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
    """
    # ImageNet mean and std for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define transformations for training, validation, and testing
    train_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.transforms.Normalize(
                                mean=[-m / s for m, s in zip(imagenet_mean, imagenet_std)],
                                std=[1 / s for s in imagenet_std]
                            )  # ImageNet stats
    ])

    test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),
                transforms.transforms.Normalize(
                                mean=[-m / s for m, s in zip(imagenet_mean, imagenet_std)],
                                std=[1 / s for s in imagenet_std]
                            ) # ImageNet stats
    ])

    # Paths to subdirectories
    train_dir = f"{main_dir}/train"
    val_dir = f"{main_dir}/val"
    test_dir = f"{main_dir}/test"

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
