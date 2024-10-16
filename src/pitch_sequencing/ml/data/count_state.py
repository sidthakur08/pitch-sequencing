import typing 

import pandas as pd
import torch

from dataclasses import dataclass
from torch.utils.data import Dataset

from pitch_sequencing.ml.tokenizers.pitch_sequence import PitchSequenceWithCountTokenizer, SeparateSequenceTokenizer
from pitch_sequencing.ml.data.last_pitch import extract_last_element_from_csv_seq
from pitch_sequencing.ml.data.sequences import SingularSequence

@dataclass
class PitchCountSequences:
    """
    pitch_seq: tokenized sequence of pitches type long. Size 1xN. Padded
    count_seq: tokenized sequence of counts type long. Size 1xN. Padded. Note: padding should be = len(padding of pitch_seq)
    src_mask: boolean tensor indicating padding of count_seq. Size 1xN.
    """
    pitch_seq: torch.Tensor
    count_seq: torch.Tensor
    src_mask: torch.Tensor

    def to(self, device: torch.device) -> 'PitchCountSequences':
        self.pitch_seq = self.pitch_seq.to(device)
        self.count_seq = self.count_seq.to(device)
        self.src_mask = self.src_mask.to(device)

        return self


# TODO(kaelen) figure out how to not write these for each type
def collate_pitch_count_seqs_and_target(batch) -> typing.Tuple[PitchCountSequences, torch.Tensor]:
    seq_data_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    pitch_seqs = torch.stack([data.pitch_seq for data in seq_data_list])
    count_seqs = torch.stack([data.count_seq for data in seq_data_list])
    src_masks = torch.stack([data.src_mask for data in seq_data_list])

    return PitchCountSequences(pitch_seqs, count_seqs, src_masks), targets

    
class LastPitchSequenceWithCountDataset(Dataset):
    """
    """
    def __init__(self, df: pd.DataFrame, seq_tokenizer: PitchSequenceWithCountTokenizer, expand_target=False, seq_df_key: str = "pitch_sequence", count_seq_df_key: str = "count_sequence") -> None:
        """
        expand_target indicates if we should give the full vocabulary output vector with the target index set to 1 (binary cross entropy). Or a tensor of size 1 with the id of our target (cross entropy loss).  
        """
        super().__init__()
        self.df = df
        self.seq_tokenizer = seq_tokenizer
        self.seq_df_key = seq_df_key
        self.count_seq_df_key = count_seq_df_key
        self.expand_target = expand_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_pitch_seq = row[self.seq_df_key]
        raw_count_seq = row[self.count_seq_df_key]
        missing_last_pitch_raw_seq, last_pitch = extract_last_element_from_csv_seq(raw_pitch_seq)

        input_seq, padding_mask = self.seq_tokenizer.tokenize(missing_last_pitch_raw_seq, raw_count_seq)
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

        target_id = self.seq_tokenizer.get_id_for_pitch(last_pitch)
        if self.expand_target:
            target = torch.zeros(self.seq_tokenizer.vocab_size())
            target[target_id] = 1
        else:
            target = torch.tensor(target_id, dtype=torch.long)

        return SingularSequence(input_seq, padding_mask), target
    
class SeparateSequencesWithCountDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_tokenizer: SeparateSequenceTokenizer, expand_target: bool, seq_df_key: str = "pitch_sequence", count_seq_df_key: str = "count_sequence") -> None:
        """
        expand_target indicates if we should give the full vocabulary output vector with the target index set to 1 (binary cross entropy). Or a tensor of size 1 with the id of our target (cross entropy loss).  
        """
        super().__init__()
        self.df = df
        self.seq_tokenizer = seq_tokenizer
        self.seq_df_key = seq_df_key
        self.count_seq_df_key = count_seq_df_key
        self.expand_target = expand_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_pitch_seq = row[self.seq_df_key]
        raw_count_seq = row[self.count_seq_df_key]
        missing_last_pitch_raw_seq, last_pitch = extract_last_element_from_csv_seq(raw_pitch_seq)

        input_seq, count_seq, padding_mask = self.seq_tokenizer.tokenize(missing_last_pitch_raw_seq, raw_count_seq)
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        count_seq = torch.tensor(count_seq, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        target_id = self.seq_tokenizer.get_id_for_pitch(last_pitch)
        if self.expand_target:
            target = torch.zeros(self.seq_tokenizer.vocab_size())
            target[target_id] = 1
        else:
            target = torch.tensor(target_id, dtype=torch.long)

        return PitchCountSequences(input_seq, count_seq, padding_mask), target
