import typing 

import pandas as pd
import torch

from dataclasses import dataclass
from torch.utils.data import Dataset

from pitch_sequencing.ml.tokenizers.pitch_arsenal import PitchArsenalLookupTable, ArsenalSequenceTokenizer
from pitch_sequencing.ml.data.last_pitch import extract_last_element_from_csv_seq

@dataclass
class SingularSequence:
    """
    src: tokenized sequence of type long. Size 1xN
    src_mask: boolean tensor indicating padding of src. Size 1xN.
    """
    src: torch.Tensor
    src_mask: torch.Tensor

    def to(self, device: torch.device) -> 'SingularSequence':
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)

        return self

    def unsqueeze(self, dim: int) -> 'SingularSequence':
        self.src = self.src.unsqueeze(dim)
        self.src_mask = self.src_mask.unsqueeze(dim)

        return self


# TODO(kaelen) figure out how to not write these for each type
def collate_interleaved_and_target(batch) -> typing.Tuple[SingularSequence, torch.Tensor]:
    seq_data_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    srcs = torch.stack([data.src for data in seq_data_list])
    src_masks = torch.stack([data.src_mask for data in seq_data_list])

    return SingularSequence(srcs, src_masks), targets

    
class PitchArsenalSequenceDataset(Dataset):
    """
    """
    def __init__(self, df: pd.DataFrame, tokenizer: ArsenalSequenceTokenizer, arsenal_lookup_table: PitchArsenalLookupTable, seq_df_key: str = "pitch_sequence", count_seq_df_key: str = "count_sequence", pitcher_id_df_key = "pitcher_id") -> None:
        super().__init__()
        self.df = df
        self.seq_tokenizer = tokenizer
        self.arsenal_lookup_table = arsenal_lookup_table
        self.seq_df_key = seq_df_key
        self.count_seq_df_key = count_seq_df_key
        self.pitcher_id_df_key = pitcher_id_df_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_pitch_seq = row[self.seq_df_key]
        raw_count_seq = row[self.count_seq_df_key]
        pitcher_id = row[self.pitcher_id_df_key]

        try:
            pitcher_arsenal = self.arsenal_lookup_table.arsenal_for_pitcher_id(pitcher_id)
        except Exception as e:
            raise ValueError(f"Failed to get pitcher arsenal for {pitcher_id}: {e}")
        
        missing_last_pitch_raw_seq, last_pitch = extract_last_element_from_csv_seq(raw_pitch_seq)

        input_seq, padding_mask = self.seq_tokenizer.tokenize(missing_last_pitch_raw_seq, raw_count_seq, pitcher_arsenal)
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

        target_id = self.seq_tokenizer.get_id_for_pitch(last_pitch)
        target = torch.tensor(target_id, dtype=torch.long)

        return SingularSequence(input_seq, padding_mask), target
