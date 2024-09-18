import typing

class HardCodedPitchSequenceTokenizer:
    def __init__(self):
        # Generated using the sequences of 
        # gs://pitch-sequencing/sequence_data/large_sequence_data_cur_opt.csv
        self._pitch_to_id: typing.Dict[str, int] = \
            {'<pad>': 0, 
            '<start>': 1, 
            '<arsenal>': 2, 
            'CB': 3, 
            'KN': 4, 
            'FC': 5, 
            'FS': 6, 
            'CH': 7, 
            'FF': 8, 
            'SL': 9, 
            'PO': 10, 
            'SI': 11, 
            'ST': 12,
            }
        
        self._pad_id = 0
        self._start_id = 1
        
        # Hardcoded to 32 for nice base 2 ness and larger than max sequence len found in above.
        self.max_sequence_len = 32
        
        # Reverse the mapping of above.
        self._id_to_pitch: typing.Dict[int, str] = {idx: pitch for pitch, idx in self._idx_to_pitch.items()} 
        

    def tokenize(self, sequence: str) -> typing.List[int]:
        """
        sequence is a CSV string of pitch strings seen in mapping table.

        If sequence is longer than max_sequence_len-1 (start token), raise error.

        returns a list of ids with a fixed start token at the beginning, followed by
             the id of each pitch found in the input sequence in the given order.
             Padded with 0s to the max_sequence_len.
        """

        tokenized_seq = [self._start_id] + [self.pitch_to_idx[pitch] for pitch in sequence.split(',')]
        if len(tokenized_seq) > self.max_sequence_len:
            raise ValueError(f"Input sequence length {len(tokenized_seq)} > {self.max_sequence_len}")
        
        if len(tokenized_seq) < self.max_sequence_len:
            tokenized_seq = tokenized_seq + [self._pad_id] * (self.max_seq_length - len(tokenized_seq))

        return tokenized_seq
    
    def get_id_for_pitch(self, pitch: str) -> int:
        if pitch not in self._pitch_to_id:
            raise KeyError(f"Pitch {pitch} not in known pitch mapping")
        
        return self._pitch_to_id[pitch]
    
    def vocab_size(self) -> int:
        return len(self._pitch_to_id)

