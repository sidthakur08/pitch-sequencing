import pandas as pd
import torch

from torch.utils.data import Dataset

from pitch_sequencing.ml.tokenizers.pitch_arsenal import PitchArsenalLookupTable, ArsenalSequenceTokenizer
from pitch_sequencing.ml.data.last_pitch import extract_last_element_from_csv_seq
from pitch_sequencing.ml.data.sequences import SingularSequence

    
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
