import typing

from itertools import zip_longest

# Generated using
# gs://pitch-sequencing/sequence_data/large_sequence_data_cur_opt.csv
ORDERED_PITCHES=['CB', 'KN', 'FC', 'FS', 'CH', 'FF', 'SL', 'PO', 'SI', 'ST']
ORDERED_COUNT=['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']

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
        self._id_to_pitch: typing.Dict[int, str] = {id: pitch for pitch, id in self._pitch_to_id.items()} 
        

    def tokenize(self, sequence: str) -> typing.List[int]:
        """
        sequence is a CSV string of pitch strings seen in mapping table.

        If sequence is longer than max_sequence_len-1 (start token), raise error.

        returns a list of ids with a fixed start token at the beginning, followed by
             the id of each pitch found in the input sequence in the given order.
             Padded with 0s to the max_sequence_len.
        """
        if len(sequence) == 0:
            raise ValueError("Given input sequence is empty")

        tokenized_seq = [self._start_id] + [self._pitch_to_id[pitch] for pitch in sequence.split(',')]

        if len(tokenized_seq) > self.max_sequence_len:
            raise ValueError(f"Input sequence length {len(tokenized_seq)} > {self.max_sequence_len}")
        
        if len(tokenized_seq) < self.max_sequence_len:
            tokenized_seq = tokenized_seq + [self._pad_id] * (self.max_sequence_len - len(tokenized_seq))

        return tokenized_seq
    
    def get_id_for_pitch(self, pitch: str) -> int:
        if pitch not in self._pitch_to_id:
            raise KeyError(f"Pitch {pitch} not in known pitch mapping")
        
        return self._pitch_to_id[pitch]
    
    def get_pitch_for_id(self, id: int) -> str:
        return self._id_to_pitch[id]
    
    def vocab_size(self) -> int:
        return len(self._pitch_to_id)

class PitchSequenceWithCountTokenizer:
    def __init__(self, pitches: typing.List[str]=ORDERED_PITCHES):
        ids = ['<pad>', '<start>', '<arsenal>']
        ids.extend(pitches)
        ids.extend(ORDERED_COUNT)

        self._pitch_to_id = {}
        for i in range(0, len(ids)):
            self._pitch_to_id[ids[i]] = i
        
        self._pad_id = 0
        self._start_id = 1
        
        self.max_sequence_len = 64
        
        # Reverse the mapping of above.
        self._id_to_pitch: typing.Dict[int, str] = {id: pitch for pitch, id in self._pitch_to_id.items()} 
    
    def padding_token_id(self) -> int:
        return self._pad_id

    def tokenize(self, pitch_sequence: str, count_sequence: str) -> typing.Tuple[typing.List[int], typing.List[bool]]:
        """
        sequence is a CSV string of pitch strings seen in mapping table.

        If sequence is longer than max_sequence_len-1 (start token), raise error.

        returns a list of ids with a fixed start token at the beginning, followed by
             the id of each pitch found in the input sequence in the given order.
             Padded with 0s to the max_sequence_len.

             and a sequence .
        """
        if len(pitch_sequence) == 0 or len(count_sequence) == 0:
            raise ValueError("Given input sequence is empty")
        
        split_pitches = pitch_sequence.split(',')
        split_counts = count_sequence.split(',')
        if len(split_pitches) != len(split_counts) - 1:
            raise ValueError(f"Sequence lengths don't match len({split_pitches}) {len(split_pitches)} != len({split_counts}) - 1{len(split_counts) - 1}")
        
        tokenized_seq = [self._start_id]
        zipped_pairs = zip_longest(split_counts, split_pitches, fillvalue=None)
        interleaved_counts_and_pitches = [item for pair in zipped_pairs for item in pair if item is not None]
        tokenized_seq.extend([self.get_id_for_pitch(item) for item in interleaved_counts_and_pitches])


        if len(tokenized_seq) > self.max_sequence_len:
            raise ValueError(f"Input sequence length {len(tokenized_seq)} > {self.max_sequence_len}")
        
        padding_mask = [False] * len(tokenized_seq)
        if len(tokenized_seq) < self.max_sequence_len:
            pad_length = (self.max_sequence_len - len(tokenized_seq))
            tokenized_seq = tokenized_seq + [self._pad_id] * pad_length
            padding_mask = padding_mask + [True] * pad_length

        #padding_mask = (tokenized_seq == self._pad_id)

        return tokenized_seq, padding_mask
    
    def get_id_for_pitch(self, pitch: str) -> int:
        if pitch not in self._pitch_to_id:
            raise KeyError(f"Pitch {pitch} not in known pitch mapping")
        
        return self._pitch_to_id[pitch]
    
    def get_pitch_for_id(self, id: int) -> str:
        return self._id_to_pitch[id]
    
    def vocab_size(self) -> int:
        return len(self._pitch_to_id)
    
class SeparateSequenceTokenizer():
    def __init__(self, pitches: typing.List[str]=ORDERED_PITCHES):
        ids = ['<pad>', '<start>', '<arsenal>']
        ids.extend(pitches)
        ids.extend(ORDERED_COUNT)

        self._pitch_to_id = {}
        for i in range(0, len(ids)):
            self._pitch_to_id[ids[i]] = i
        
        self._pad_id = 0
        self._start_id = 1
        
        self.max_sequence_len = 32
        
        # Reverse the mapping of above.
        self._id_to_pitch: typing.Dict[int, str] = {id: pitch for pitch, id in self._pitch_to_id.items()} 

    def _tokenize_and_pad_sequence(self, sequence: typing.List[str]) -> typing.Tuple[typing.List[int], typing.List[bool]]:
        tokenized_seq = [self._pitch_to_id[item] for item in sequence]

        if len(tokenized_seq) > self.max_sequence_len:
            raise ValueError(f"Input sequence length {len(tokenized_seq)} > {self.max_sequence_len}")

        padding_mask = [False] * len(tokenized_seq)
        if len(tokenized_seq) < self.max_sequence_len:
            pad_length = (self.max_sequence_len - len(tokenized_seq))
            tokenized_seq = tokenized_seq + [self._pad_id] * pad_length
            padding_mask = padding_mask + [True] * pad_length

        return tokenized_seq, padding_mask

    def tokenize(self, pitch_sequence: str, count_sequence: str) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[bool]]:
        if len(pitch_sequence) == 0 or len(count_sequence) == 0:
            raise ValueError("Given input sequence is empty")
        
        split_pitches = pitch_sequence.split(',')
        split_counts = count_sequence.split(',')
        if len(split_pitches) != len(split_counts) - 1:
            raise ValueError(f"Sequence lengths don't match len({split_pitches}) {len(split_pitches)} != len({split_counts}) - 1{len(split_counts) - 1}")
        
        tokenized_pitches, pitches_padding_mask = self._tokenize_and_pad_sequence(split_pitches)
        tokenized_counts, counts_padding_mask = self._tokenize_and_pad_sequence(split_counts)

        return tokenized_pitches, tokenized_counts, counts_padding_mask
    
    def get_id_for_pitch(self, pitch: str) -> int:
        if pitch not in self._pitch_to_id:
            raise KeyError(f"Pitch {pitch} not in known pitch mapping")
        
        return self._pitch_to_id[pitch]
    
    def get_pitch_for_id(self, id: int) -> str:
        return self._id_to_pitch[id]
    
    def vocab_size(self) -> int:
        return len(self._pitch_to_id)
    