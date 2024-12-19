# trainer.py

import torch
import torch.nn as nn
import time
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 early_stopping_patience: int = 10):
        """
        Initialize trainer with model and training parameters
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def prepare_data(self,
                     X: torch.Tensor,
                     y: torch.Tensor,
                     val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )

        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              criterion: nn.Module = None) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and regularization
        """
        if criterion is None:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Add L2 regularization
                l2_lambda = 0.01
                l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss = loss + l2_lambda * l2_norm

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_steps += 1

            # Calculate average losses
            avg_train_loss = train_loss / train_steps
            avg_val_loss = val_loss / val_steps

            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        return self.history