import typing

import pandas as pd

from itertools import zip_longest

# Generated using
# gs://pitch-sequencing/sequence_data/large_sequence_data_cur_opt.csv
ORDERED_PITCHES=['CB', 'KN', 'FC', 'FS', 'CH', 'FF', 'SL', 'PO', 'SI', 'ST']
ORDERED_COUNT=['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']

class PitchArsenalLookupTable:
    def __init__(self, arsenal_mapping_df: pd.DataFrame):
        self.max_arsenal_size = arsenal_mapping_df['arsenal_size'].max()
        self.pitcher_arsenal_table = arsenal_mapping_df.set_index('pitcher')['pitch_arsenal_csv'].to_dict()

    def arsenal_for_pitcher_id(self, pitcher_id: int) -> typing.List[str]:
        return self.pitcher_arsenal_table[pitcher_id]

class ArsenalSequenceTokenizer:
    def __init__(self, max_arsenal_size: int = 10, max_pitch_count_seq_len: int = 63):
        ids = ['<pad>', '<start>', '<arsenal>', '<seq_start>']
        ids.extend(ORDERED_PITCHES)
        ids.extend(ORDERED_COUNT)


        self._pitch_to_id = {}
        for i in range(0, len(ids)):
            self._pitch_to_id[ids[i]] = i
        
        self._pad_id = 0
        self._start_id = 1
        self._arsenal_start_id = 2
        self._seq_start_id = 3
        
        # Add one for seq_start token
        self.max_pitch_count_seq_length = max_pitch_count_seq_len + 1
        # Add one for arsenal start token
        self.max_arsenal_length = max_arsenal_size + 1
        # Add one for full sequence start token. This allows for padding to happen at component level.
        self.max_sequence_length = self.max_pitch_count_seq_length + self.max_arsenal_length + 1
        
        # Reverse the mapping of above.
        self._id_to_pitch: typing.Dict[int, str] = {id: pitch for pitch, id in self._pitch_to_id.items()} 
    
    def padding_token_id(self) -> int:
        return self._pad_id
    
    def _tokenize_seq(self, raw_sequence: typing.List[str], start_id: int, max_encoded_seq_len: int) -> typing.List[int]:
        encoded_seq = [start_id]
        encoded_seq = encoded_seq + [self.get_id_for_pitch(item) for item in raw_sequence]

        if len(encoded_seq) > max_encoded_seq_len:
            raise ValueError(f"Encoded sequence {len(encoded_seq)} length larger than allowed {max_encoded_seq_len}")
        
        if len(encoded_seq) < max_encoded_seq_len:
            pad_length = (max_encoded_seq_len - len(encoded_seq))
            encoded_seq = encoded_seq + [self._pad_id] * pad_length

        return encoded_seq


    def tokenize(self, pitch_sequence: str, count_sequence: str, pitch_arsenal: str) -> typing.Tuple[typing.List[int], typing.List[bool]]:
        """
        sequence is a CSV string of pitch strings seen in mapping table.

        If sequence is longer than max_sequence_len-1 (start token), raise error.

        returns a list of ids with a fixed start token at the beginning, followed by
             the id of each pitch found in the input sequence in the given order.
             Padded with 0s to the max_sequence_len.

             and a sequence .
        """

        try:
            encoded_arsenal_seq = self._tokenize_seq(pitch_arsenal.split(','), self._arsenal_start_id, self.max_arsenal_length)
        except ValueError as e:
            raise ValueError(f"Failed encoding arsenl sequence {pitch_arsenal}: {e}")


        ### Do Pitch and Count sequence generation
        # TODO(kaelen): Factor this out and reuse with other count sequence tokenizer
        if len(pitch_sequence) == 0 or len(count_sequence) == 0:
            raise ValueError("Given input sequence is empty")
        
        split_pitches = pitch_sequence.split(',')
        split_counts = count_sequence.split(',')
        if len(split_pitches) != len(split_counts) - 1:
            raise ValueError(f"Sequence lengths don't match len({split_pitches}) {len(split_pitches)} != len({split_counts}) - 1{len(split_counts) - 1}")
        
        # Interleave counts and pitches with count starting first.
        zipped_pairs = zip_longest(split_counts, split_pitches, fillvalue=None)
        interleaved_counts_and_pitches = [item for pair in zipped_pairs for item in pair if item is not None]
        try:
            tokenized_pitch_count_seq = self._tokenize_seq(interleaved_counts_and_pitches, self._seq_start_id, self.max_pitch_count_seq_length)
        except ValueError as e:
            raise ValueError(f"Failed encoding pitch and count sequences: {e}")
        
        # Don't need to check sizes here since they're enforced at the component level.
        tokenized_seq = [self._start_id] + encoded_arsenal_seq + tokenized_pitch_count_seq
        padding_mask = [token == self._pad_id for token in tokenized_seq]

        return tokenized_seq, padding_mask
    
    def get_id_for_pitch(self, pitch: str) -> int:
        if pitch not in self._pitch_to_id:
            raise KeyError(f"Pitch {pitch} not in known pitch mapping")
        
        return self._pitch_to_id[pitch]
    
    def get_pitch_for_id(self, id: int) -> str:
        return self._id_to_pitch[id]
    
    def vocab_size(self) -> int:
        return len(self._pitch_to_id)