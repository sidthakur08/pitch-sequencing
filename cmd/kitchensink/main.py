import argparse
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import asdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pitch_sequencing.ml.tokenizers.vocab as vocab
import pitch_sequencing.ml.data.generators as gen

from pitch_sequencing.ml.data.sequences import PitchSequenceDataset, CSVSequenceDataGenPlan, collate_interleaved_and_target, 
from pitch_sequencing.ml.tokenizers.pitch_arsenal import PitchArsenalLookupTable
from pitch_sequencing.ml.tokenizers.pitch_sequence import PitchSequenceTokenizer, SequenceInfo, SequenceID
from pitch_sequencing.ml.models.last_pitch import LastPitchTransformerModel
from pitch_sequencing.io.join import join_paths
from pitch_sequencing.io.gcs import save_model_to_gcs



def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_criterion: nn.Module, num_epochs: int, lr: float, device: torch.device, output_directory: str, logging_directory: str, output_len: int) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # SummaryWriter for Tensorboard logging metrics
    summary_writer = SummaryWriter(log_dir=logging_directory)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()

            input_data, target = [b.to(device) for b in batch]
            output = model(**asdict(input_data))

            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        summary_writer.add_scalar("train/avg_loss", train_loss/len(train_loader), epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_data, target = [b.to(device) for b in batch]
                output = model(**asdict(input_data))
                loss = loss_criterion(output, target)
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

    parser.add_argument("--input_train_path", type=str, required=True, help="Input Path to training data.")
    parser.add_argument("--input_validation_path", type=str, required=True, help="Validation data input path")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory path for all output artifacts from training job")
    parser.add_argument("--logging_directory", type=str, required=True, help="Base directory where all logging files will be stored, including tensorboard")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")

    parser.add_argument("--arsenal_lookup_table_path", type=str, required=True, help="Path to a csv file that contains the pitcher aresnal information")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(f"Reading data from {args.input_train_path}")
    train_df = pd.read_csv(args.input_train_path)
    print(f"Reading data from {args.input_validation_path}")
    validation_df = pd.read_csv(args.input_validation_path)
    print(train_df.shape)
    print(validation_df.shape)

    print(f"Reading Pitcher arsenal data from {args.arsenal_lookup_table_path}")
    arsenal_df = pd.read_csv(args.arsenal_lookup_table_path)
    arsenal_lookup_table = PitchArsenalLookupTable(arsenal_df)
    print(f"Successfully loaded PItcher arsenal data")

    sequential_sequence_infos = [
        SequenceInfo(SequenceID.ARSENAL, arsenal_lookup_table.max_arsenal_size, vocab_ids=[vocab.VocabID.PITCHES]),
        SequenceInfo(SequenceID.HANDEDNESS, 2, vocab_ids=[vocab.VocabID.HANDEDNESS]),
        SequenceInfo(SequenceID.ON_BASE, 3, vocab_ids=[vocab.VocabID.BOOLEAN]),
    ]
    sequential_sequence_gen_plans = [
        CSVSequenceDataGenPlan(SequenceID.ARSENAL, gen.ArsenalCSVGenerator(arsenal_lookup_table)),
        CSVSequenceDataGenPlan(SequenceID.HANDEDNESS, gen.HandednessCSVGenerator()),
        CSVSequenceDataGenPlan(SequenceID.ON_BASE, gen.OnBaseCSVGenerator()),
    ]

    # Hardcode 63 for now.
    interleaved_sequence_infos = SequenceInfo(SequenceID.INTERLEAVED, 63, vocab_ids=[vocab.VocabID.PITCHES, vocab.VocabID.COUNTS])
    interleaved_sequence_gen_plans = [
        CSVSequenceDataGenPlan(SequenceID.INTERLEAVED, gen.DirectCSVLookupGenerator('count_sequence')),
        CSVSequenceDataGenPlan(SequenceID.PITCHES, gen.DirectCSVLookupGenerator('input_pitch_sequence')),
    ]

    tokenizer = PitchSequenceTokenizer(sequential_sequence_infos, interleaved_sequence_infos, [vocab.PITCH_VOCAB, vocab.HANDEDNESS_VOCAB, vocab.BOOLEAN_VOCAB, vocab.COUNT_VOCAB])
    train_dataset = PitchSequenceDataset(train_df, tokenizer, sequential_sequence_gen_plans, interleaved_sequence_gen_plans, target_df_key='target_pitch')
    validation_dataset = PitchSequenceDataset(validation_df, tokenizer, sequential_sequence_gen_plans, interleaved_sequence_gen_plans, target_df_key='target_pitch')
    model = LastPitchTransformerModel(tokenizer.vocab_size(), d_model=64, nhead=4, num_layers=2)
    collate_fn = collate_interleaved_and_target
    loss = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # Train model
    print("Starting Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.is_available()}")
    _ = train_model(model, train_loader, validation_loader, loss, num_epochs=args.num_epochs, lr=args.learning_rate, device=device, output_directory=args.output_directory, logging_directory=args.logging_directory, output_len=tokenizer.vocab_size())

    print(f"Done training model. Output can be found at {args.output_directory}")
