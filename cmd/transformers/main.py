import argparse
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gcsfs

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict

from pitch_sequencing.io.join import join_paths


class PitchDataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.pitch_to_idx = {'<pad>': 0, '<start>': 1, '<arsenal>': 2}
        self.idx_to_pitch = {0: '<pad>', 1: '<start>', 2: '<arsenal>'}
        self.pitcher_arsenals = defaultdict(set)
        self.max_seq_length = 0
        self.max_arsenal_length = 8

    def preprocess(self):
        # Create pitch type vocabulary
        unique_pitches = set()
        for seq in self.df['Pitch Sequence']:
            unique_pitches.update(seq.split(','))
        
        for pitch in unique_pitches:
            if pitch not in self.pitch_to_idx:
                idx = len(self.pitch_to_idx)
                self.pitch_to_idx[pitch] = idx
                self.idx_to_pitch[idx] = pitch

        # Create pitcher arsenals
        for _, row in self.df.iterrows():
            pitcher_id = row['Pitcher ID']
            pitches = row['Pitch Sequence'].split(',')
            self.pitcher_arsenals[pitcher_id].update(pitches)

        # Find max sequence length (including <start> token)
        # TODO(kaelen): temporarily added 9 padding spaces for arsenal length for now. Think of better way to do this.
        self.max_seq_length = max(len(seq.split(',')) for seq in self.df['Pitch Sequence']) + 10

    def encode_input(self, sequence, pitcher_id):
        encoded_arsenals = self.encode_arsenal_for_pitcher(pitcher_id)
        encoded_sequence = self.encode_sequence(sequence)
        return encoded_arsenals + encoded_sequence
    
    def encode_arsenal_for_pitcher(self, pitcher_id):
        arsenal = self.pitcher_arsenals[pitcher_id]
        arsenal_ids = [self.pitch_to_idx[pitch] for pitch in arsenal]
        # [<arsenal>, <pitches in arsenal...>, <pad (if needed)>]
        encoded_arsenals = [2] + sorted(arsenal_ids) + [0] * (self.max_arsenal_length - len(arsenal))
        return encoded_arsenals

    def encode_sequence(self, sequence):
        return [1] + [self.pitch_to_idx[pitch] for pitch in sequence.split(',')]

    def pad_sequence(self, sequence):
        padded = sequence + [0] * (self.max_seq_length - len(sequence))
        return padded[:self.max_seq_length]

    def get_pitcher_arsenal_mask(self, pitcher_id):
        arsenal = self.pitcher_arsenals[pitcher_id]
        mask = [1 if pitch in arsenal or idx < 3 else 0 for idx, pitch in self.idx_to_pitch.items()]
        return torch.tensor(mask, dtype=torch.float)
    

class PitchSequenceDataset(Dataset):
    def __init__(self, df, preprocessor):
        self.df = df
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = self.preprocessor.encode_sequence(row['Pitch Sequence'])
        
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target = torch.tensor(sequence[-1], dtype=torch.long)
        
        padded_input = self.preprocessor.pad_sequence(input_seq.tolist())
        input_seq = torch.tensor(padded_input, dtype=torch.long)

        return input_seq, target
    

class PitchSequenceAndPitcherDataset(Dataset):
    def __init__(self, df, preprocessor):
        self.df = df
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pitcher_id = row['Pitcher ID']
        raw_sequence = row['Pitch Sequence']
        encoded_input = self.preprocessor.encode_input(raw_sequence, pitcher_id)
        
        # Strip off the last pitch from our sequence and make it the target pitch we want to predict. 
        input_seq = torch.tensor(encoded_input[:-1], dtype=torch.long)
        target = torch.tensor(encoded_input[-1], dtype=torch.long)
        
        padded_input = self.preprocessor.pad_sequence(input_seq.tolist())
        input_seq = torch.tensor(padded_input, dtype=torch.long)
        
        return input_seq, target
    
class PitchTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super(PitchTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)  # Assuming max sequence length < 1000
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        pos = torch.arange(0, src.size(1), dtype=torch.long, device=src.device).unsqueeze(0)
        src = src + self.pos_encoder(pos)
        output = self.transformer(src, src_key_padding_mask=src_mask)
        return self.fc(output[:, -1, :])  # Only use the last position for prediction



def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, lr: float, device: torch.device, output_directory: str) -> nn.Module:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Don't use arsenal mask for now.
                input_seq, target = [b.to(device) for b in batch]
                output = model(input_seq)
                loss = criterion(output, target)
                val_loss += loss.item()

        elapsed_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}. Epoch Time {elapsed_time:.2f}sec")

    saved_path = save_model_to_gcs(model, join_paths(output_directory, "final", "model.pth"))
    print(f"Saved final model to {saved_path}")

    return model


def join_paths(*paths: str) -> str:
    """
        joins the path of two path objects. strips any excess '/' chars from inputs before joining with '/'
    """
    
    cleaned_paths = [path.strip('/') for path in paths]
    return "/".join(cleaned_paths)

def save_model_to_gcs(model: nn.Module, gcs_path: str) -> str:
    fs = gcsfs.GCSFileSystem(project="pitch-sequencing")

    with fs.open(gcs_path, 'wb') as f:
        torch.save(model.state_dict(), f)

    return gcs_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ML model.")

    parser.add_argument("--input_path", type=str, required=True, help="Input Path to training data.")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory path for all output artifacts from training job")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Inputa Args: {args}")

    input_data_path = args.input_path
    #"gs://pitch-sequencing/sequence_data/large_sequence_data_cur_opt.csv"
    print(f"Reading data from {input_data_path}")
    df = pd.read_csv(input_data_path)


    print(join_paths("hello", "world", "goodbye", "world"))

    # Preprocess data
    preprocessor = PitchDataPreprocessor(df)
    print("Running preprocessing")
    preprocessor.preprocess()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(train_df.shape)
    print(test_df.shape)

    # Create datasets and dataloaders
    train_dataset = PitchSequenceDataset(train_df, preprocessor)
    test_dataset = PitchSequenceDataset(test_df, preprocessor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    vocab_size = len(preprocessor.pitch_to_idx)
    model = PitchTransformer(vocab_size, d_model=64, nhead=4, num_layers=2)

    # Train model
    print("Starting Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.is_available()}")
    _ = train_model(model, train_loader, test_loader, num_epochs=args.num_epochs, lr=args.learning_rate, device=device, output_directory=args.output_directory)

    print(f"Done training model. Output can be found at {args.output_directory}")
