import torch
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from torch.utils.data import Dataset

import torch.nn as nn

from train_swept_models import (
    BETHOVENConfig,
    BEETHOVEN,
    train_epoch,
    validate,
    setup_optimizer,
    setup_scheduler
)


def train_beethoven_malevis(
    data_dir: str,
    output_dir: str = "./checkpoints",
    config: dict = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train BEETHOVEN model on Malevis dataset.
    
    Args:
        data_dir: Path to Malevis dataset
        output_dir: Directory to save checkpoints
        config: BEETHOVEN configuration dict
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    # Initialize wandb
    wandb.init(project="beethoven-malevis", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        **config
    })
    
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Malevis dataset
    # TODO: Implement MalevisDataset class or import it
    # train_dataset = MalevisDataset(data_dir, split="train")
    # val_dataset = MalevisDataset(data_dir, split="val")
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize BEETHOVEN model
    beethoven_config = BETHOVENConfig(**config) if config else BETHOVENConfig()
    model = BEETHOVEN(beethoven_config).to(device)
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, learning_rate)
    scheduler = setup_scheduler(optimizer, epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device)
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_path / f"beethoven_malevis_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': beethoven_config
            }, checkpoint_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path / f"beethoven_malevis_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': beethoven_config
            }, checkpoint_path)
    
    wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    # Example configuration for BEETHOVEN
    config = {
        "input_dim": 2381,  # Adjust based on Malevis features
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.1,
        "num_classes": 25  # Adjust based on Malevis malware families
    }
    
    train_beethoven_malevis(
        data_dir="./malevis_data",
        output_dir="./checkpoints",
        config=config,
        epochs=50,
        batch_size=32,
        learning_rate=1e-4
    )