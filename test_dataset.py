from data import dataset
from config import Config
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = Config()
    # Create an instance of the dataset
    train_dataset, val_dataset, test_dataset = dataset.prepare_datasets(config)
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )

    print(train_loader, val_loader, test_loader)
    # Print the first 5 samples from the dataset
    
    n = 10 #кол-во изображений для отображения
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    images, batch, labels = next(iter(train_loader))

    for i in range(n):
        img = batch[i].detach().cpu()

        # [C, H, W] -> [H, W, C]
        if img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)

        axes[i].imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        axes[i].set_title(f"AI probability: {labels[i].item()}")
        axes[i].axis("off")

    plt.show()

