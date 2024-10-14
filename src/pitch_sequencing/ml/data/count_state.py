import pandas as pd
import torch

from torch.utils.data import Dataset

from pitch_sequencing.ml.tokenizers.pitch_sequence import PitchSequenceWithCountTokenizer
from pitch_sequencing.ml.data.last_pitch import extract_last_element_from_csv_seq
 

class LastPitchSequenceWithCountDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_tokenizer: PitchSequenceWithCountTokenizer, seq_df_key: str = "pitch_sequence", count_seq_df_key: str = "count_sequence") -> None:
        super().__init__()
        self.df = df
        self.seq_tokenizer = seq_tokenizer
        self.seq_df_key = seq_df_key
        self.count_seq_df_key = count_seq_df_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_pitch_seq = row[self.seq_df_key]
        raw_count_seq = row[self.count_seq_df_key]
        missing_last_pitch_raw_seq, last_pitch = extract_last_element_from_csv_seq(raw_pitch_seq)
        #missing_last_count_raw_seq, _ = extract_last_element_from_csv_seq(raw_count_seq)

        input_seq, padding_mask = self.seq_tokenizer.tokenize(missing_last_pitch_raw_seq, raw_count_seq)
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        target = torch.tensor(self.seq_tokenizer.get_id_for_pitch(last_pitch), dtype=torch.long)

        return input_seq, padding_mask, target
