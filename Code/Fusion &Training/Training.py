"""
Software vulnerability Detection System using Source Code and Binary Features

This system combines:
1. Source code analysis using UniXcoder
2. Binary executable analysis using Vision Transformer
3. Multi-component classifier architecture evaluation

Author: [Wang Qi]
"""

# Standard library imports
import os
import random
import logging
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                           recall_score, f1_score, confusion_matrix)

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Custom module imports
from source_code_feature_extraction import (
    read_cpp_files_and_labels,
    convert_code_to_vector
)
from executable_feature_extraction import binary_to_grayscale_and_infer

# Constants
TARGET_DIMENSION = 512
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
LOG_DIR = "logs"
MODEL_SAVE_DIR = "saved_models"
EMBED_DIM = 512

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("malware_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# Data structures
@dataclass
class ModelConfig:
    name: str
    transformer_depth: int = 2
    transformer_heads: int = 6

@dataclass
class TrainingResults:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    train_loss_history: List[float]
    val_loss_history: List[float]

# Neural Network Components -----------------------------------------------------

class CLSTransformerLayer(nn.Module):
    """Custom Transformer layer focusing on CLS token processing"""
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = x[:, 0:1, :]
        other_tokens = x[:, 1:, :]

        cls_token = self.norm(cls_token)
        other_tokens = self.norm(other_tokens)

        attn_output, _ = self.attention(
            query=cls_token,
            key=other_tokens,
            value=other_tokens
        )
        cls_token = cls_token + attn_output
        cls_token = cls_token + self.mlp(self.norm(cls_token))

        return torch.cat([cls_token, other_tokens], dim=1)

class CLSTransformer(nn.Module):
    """Stack of CLSTransformer layers"""
    def __init__(self, dim: int, depth: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CLSTransformerLayer(dim, heads, dropout) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class BinaryClassifier(nn.Module):
    """Complete malware classifier with all components"""
    def __init__(self,
                 input_dim: int,
                 transformer_depth: int = 2,
                 transformer_heads: int = 6):
        super().__init__()
        self.embed_dim = EMBED_DIM

        # Projection layer
        self.proj = nn.Linear(input_dim, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim)

        # Transformer component
        self.transformer = CLSTransformer(
            dim=self.embed_dim,
            depth=transformer_depth,
            heads=transformer_heads
        )

        # Linear activation component
        self.linear_act = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU()
        )

        # Normalization component
        self.norm = nn.LayerNorm(self.embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to embedding dimension
        x = self.proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply all components
        x = self.transformer(x)
        x = self.linear_act(x)
        x = self.norm(x)

        # Classification using CLS token
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)

class Trainer:
    """Handles model training and evaluation"""
    def __init__(self, model: nn.Module, device: torch.device, model_name: str):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.writer = SummaryWriter(os.path.join(LOG_DIR, model_name))

        # Create directories
        os.makedirs(os.path.join(LOG_DIR, model_name), exist_ok=True)
        os.makedirs(os.path.join(MODEL_SAVE_DIR, model_name), exist_ok=True)

    def train(self, train_loader: DataLoader, test_loader: DataLoader) -> TrainingResults:
        train_loss_history = []
        val_loss_history = []

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Calculate validation loss
            val_loss, _ = self.evaluate(test_loader)
            avg_train_loss = running_loss / len(train_loader)

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(val_loss)

            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")

            # Save model checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(MODEL_SAVE_DIR, self.model_name, f"epoch_{epoch + 1}.pt")
                )

        # Final evaluation
        _, metrics = self.evaluate(test_loader, final=True)

        return TrainingResults(
            model_name=self.model_name,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            confusion_matrix=metrics['confusion_matrix'],
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history
        )

    def evaluate(self, test_loader: DataLoader, final: bool = False) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(test_loader)

        if final:
            metrics = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'precision': precision_score(all_labels, all_preds),
                'recall': recall_score(all_labels, all_preds),
                'f1': f1_score(all_labels, all_preds),
                'confusion_matrix': confusion_matrix(all_labels, all_preds)
            }
            return avg_loss, metrics

        return avg_loss, {}

def prepare_data(
    code_dir: str,
    binary_dir: str,
    output_dir: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare and combine source code and binary features"""
    logger.info("Loading and processing source code files...")
    code_files, labels, filenames = read_cpp_files_and_labels(code_dir)
    code_vectors = convert_code_to_vector(code_files)

    logger.info("Processing binary executables...")
    image_tokens, binary_filenames = binary_to_grayscale_and_infer(
        binary_folder=binary_dir,
        output_folder=output_dir
    )

    logger.info("Combining features...")
    # Create mapping from filename prefix to code vector
    code_vectors_dict = {
        filename.split('.')[0]: (vector, filename)
        for vector, filename in zip(code_vectors, filenames)
    }

    combined_tokens = []
    combined_labels = []

    for img_token, img_filename in zip(image_tokens, binary_filenames):
        img_prefix = img_filename.split('.')[0]
        if img_prefix in code_vectors_dict:
            code_vector, _ = code_vectors_dict[img_prefix]
            img_token[0] = code_vector  # Replace CLS token with code vector
            combined_tokens.append(img_token)
            combined_labels.append(int(img_filename.split('_')[0]))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(combined_tokens),
        np.array(combined_labels),
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=combined_labels
    )

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def create_data_loaders(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders"""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    return train_loader, test_loader

def evaluate_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device
) -> TrainingResults:
    """Train and evaluate the model"""
    logger.info("\nTraining complete model with all components")

    # Initialize model
    model = BinaryClassifier(
        input_dim=TARGET_DIMENSION,
        transformer_depth=2,
        transformer_heads=6
    ).to(device)

    # Train and evaluate
    trainer = Trainer(model, device, "CompleteModel")
    result = trainer.train(train_loader, test_loader)

    # Plot training curves
    plt.figure()
    plt.plot(result.train_loss_history, label='Train Loss')
    plt.plot(result.val_loss_history, label='Validation Loss')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, 'CompleteModel', 'training_curve.png'))
    plt.close()

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(result.confusion_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(LOG_DIR, 'CompleteModel', 'confusion_matrix.png'))
    plt.close()

    return result

def main():
    """Main execution pipeline"""
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Path configuration
    code_dir = "/path/to/cpp/files"
    binary_dir = "/path/to/binaries"
    output_dir = "/path/to/save/images"

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        code_dir=code_dir,
        binary_dir=binary_dir,
        output_dir=output_dir
    )

    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)

    # Train and evaluate model
    result = evaluate_model(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    # Display results
    logger.info("\nModel Results:")
    logger.info(f"Accuracy: {result.accuracy:.4f}")
    logger.info(f"Precision: {result.precision:.4f}")
    logger.info(f"Recall: {result.recall:.4f}")
    logger.info(f"F1 Score: {result.f1:.4f}")
    logger.info("Training completed and results saved")

if __name__ == "__main__":
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    try:
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise