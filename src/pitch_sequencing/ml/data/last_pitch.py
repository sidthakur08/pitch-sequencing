import pandas as pd
import torch

from torch.utils.data import Dataset

from pitch_sequencing.ml.tokenizers.pitch_sequence import HardCodedPitchSequenceTokenizer

def extract_last_pitch_from_csv_seq(raw_csv_seq: str) -> tuple[str, str]:
    """
        extracts the last pitch from a csv seq. 
        Returns the csv sequence missing the last pitch and the last pitch extracted.
    """
    parsed_seq = raw_csv_seq.split(',')
    last_pitch = parsed_seq.pop()

    missing_last_pitch_raw_seq = ",".join(parsed_seq)

    return missing_last_pitch_raw_seq, last_pitch 

class LastPitchSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_tokenizer: HardCodedPitchSequenceTokenizer, seq_df_key: str = "Pitch Sequence") -> None:
        super().__init__()
        self.df = df
        self.seq_tokenizer = seq_tokenizer
        self.seq_df_key = seq_df_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_seq = row[self.seq_df_key]
        missing_last_pitch_raw_seq, last_pitch = extract_last_pitch_from_csv_seq(raw_seq)

        input_seq = torch.tensor(self.seq_tokenizer.tokenize(missing_last_pitch_raw_seq), dtype=torch.long)
        target = torch.tensor(self.seq_tokenizer.get_id_for_pitch(last_pitch), dtype=torch.long)

        return input_seq, target