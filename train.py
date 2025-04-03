import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from model import TextGenGRU  # Import your existing model
import time

def train_model():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 20
    learning_rate = 3e-4

    # Load processed data
    data = torch.load("processed_data.pt")
    train_data = data["train"]
    val_data = data["val"]
    word2idx = data["word2idx"]
    
    # Create DataLoaders (modified for your model's input format)
    train_dataset = TensorDataset(train_data, torch.roll(train_data, -1, 1))
    val_dataset = TensorDataset(val_data, torch.roll(val_data, -1, 1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model from your model.py
    model = TextGenGRU(
        vocab_size=len(word2idx),
        emb_dim=256,
        hid_dim=512,
        num_layers=2
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        start_time = time.time()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _ = model(inputs.permute(1, 0))  # (seq_len, batch, vocab)
            
            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                targets.T.contiguous().view(-1)
            )
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs.permute(1, 0))
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    targets.T.contiguous().view(-1)
                )
                epoch_val_loss += loss.item()

        # Calculate metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved new best model with val loss: {best_val_loss:.4f}")

        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print("-" * 50)

        # Early stopping check
        if avg_val_loss > 2 * avg_train_loss:
            print("Stopping early due to overfitting!")
            break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()