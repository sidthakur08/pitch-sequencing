import argparse
import time

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pitch_sequencing.ml.data.last_pitch import LastPitchSequenceDataset
from pitch_sequencing.ml.tokenizers.pitch_sequence import HardCodedPitchSequenceTokenizer
from pitch_sequencing.ml.models.last_pitch import LastPitchTransformerModel
from pitch_sequencing.io.join import join_paths
from pitch_sequencing.io.gcs import save_model_to_gcs
from pitch_sequencing.io.args.training import add_common_training_args


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, lr: float, device: torch.device, output_directory: str, logging_directory: str) -> nn.Module:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # SummaryWriter for Tensorboard logging metrics
    summary_writer = SummaryWriter(log_dir=logging_directory)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_start = time.time()
        for batch in train_loader:
            # Don't use arsenal mask for now.
            input_seq, target = [b.to(device) for b in batch]
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        summary_writer.add_scalar("train/avg_loss", train_loss/len(train_loader), epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Don't use arsenal mask for now.
                input_seq, target = [b.to(device) for b in batch]
                output = model(input_seq)
                loss = criterion(output, target)
                val_loss += loss.item()

        summary_writer.add_scalar("val/avg_loss", val_loss/len(val_loader), epoch)

        elapsed_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}. Epoch Time {elapsed_time:.2f}sec")
        summary_writer.flush()

    saved_path = save_model_to_gcs(model, join_paths(output_directory, "final", "model.pth"))
    print(f"Saved final model to {saved_path}")
    summary_writer.close()

    return model



def parse_args():
    parser = argparse.ArgumentParser(description="Train a ML model.")
    add_common_training_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Reading data from {args.input_train_path}")
    train_df = pd.read_csv(args.input_train_path)

    print(f"Reading data from {args.input_validation_path}")
    validation_df = pd.read_csv(args.input_validation_path)
    
    print(train_df.shape)
    print(validation_df.shape)

    tokenizer = HardCodedPitchSequenceTokenizer()

    # Create datasets and dataloaders
    train_dataset = LastPitchSequenceDataset(train_df, tokenizer)
    validation_dataset = LastPitchSequenceDataset(validation_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    model = LastPitchTransformerModel(tokenizer.vocab_size(), d_model=64, nhead=4, num_layers=2)

    # Train model
    print("Starting Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.is_available()}")
    _ = train_model(model, train_loader, validation_loader, num_epochs=args.num_epochs, lr=args.learning_rate, device=device, output_directory=args.output_directory, logging_directory=args.logging_directory)

    print(f"Done training model. Output can be found at {args.output_directory}")
