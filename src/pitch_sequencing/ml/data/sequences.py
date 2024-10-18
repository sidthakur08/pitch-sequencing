import typing

import pandas as pd
import torch

from torch.utils.data import Dataset
from dataclasses import dataclass

from pitch_sequencing.ml.data.generators import CSVSequenceGenerator
from pitch_sequencing.ml.tokenizers.pitch_sequence import PitchSequenceTokenizer, SequenceID, CSVSequenceInput

@dataclass
class CSVSequenceDataGenPlan:
    seq_id: SequenceID
    generator: CSVSequenceGenerator

class PitchSequenceDataset(Dataset):
    """
    Ordering matters for plans! Should be 1:1 correspondence with the tokenizers expected input. 
    TODO(kaelen): have tokenizer be able to tell us the plan ordering it expects.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: PitchSequenceTokenizer, 
                 sequential_input_gen_plan: typing.List[CSVSequenceDataGenPlan], 
                 interleave_input_gen_plan: typing.List[CSVSequenceDataGenPlan],
                 target_df_key: str,
        ):

        self.df = df
        self.tokenizer = tokenizer
        self.sequential_input_generation_plan = sequential_input_gen_plan
        self.interleave_input_generation_plan = interleave_input_gen_plan
        self.target_df_key = target_df_key


    def __len__(self):
        return len(self.df)
    
    def __get__(self, idx):
        row = self.iloc(idx)

        sequential_inputs = []
        for plan in self.sequential_input_generation_plan:
            generated_csv_input = plan.generator.generate_csv_sequence_from_df_row(row)
            sequential_inputs.append(CSVSequenceInput(plan.seq_id, generated_csv_input))
        
        inputs_to_interleave = []
        for plan in self.sequential_input_generation_plan:
            generated_csv_input = plan.generator.generate_csv_sequence_from_df_row(row)
            inputs_to_interleave.append(generated_csv_input)

        try:
            encoded_sequence, padding_mask = self.tokenizer.tokenize(sequential_inputs, inputs_to_interleave)
        except Exception as e:
            raise ValueError(f"Failed to tokenize sequence for {idx} {e}")
        
        target_pitch = self.df[self.target_df_key]
        target_id = self.tokenizer.get_id_for_token(target_pitch)

        input_seq = torch.tensor(encoded_sequence, dtype=torch.long)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        target = torch.tensor(target_id, dtype=torch.long)

        return input_seq, padding_mask, target
