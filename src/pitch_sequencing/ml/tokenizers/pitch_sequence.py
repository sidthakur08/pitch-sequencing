import typing

from dataclasses import dataclass
from itertools import zip_longest

import pitch_sequencing.ml.tokenizers.vocab as vocab

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

def encode_sequence(sequence: typing.List[str], id_mapping_table: typing.Dict[str, int], max_encoded_seq_len: int, start_id: int, padding_id: int = 0) -> typing.List[int]:
    encoded_sequence = [start_id]
    encoded_sequence = encode_sequence + [id_mapping_table[item] for item in sequence]

    if len(encoded_sequence) > max_encoded_seq_len:
        raise ValueError(f"Encoded Sequence len {len(encoded_sequence)} > {max_encoded_seq_len} for f{sequence} + start token")
    
    if len(encoded_sequence) < max_encoded_seq_len:
        pad_length = (max_encoded_seq_len - len(encoded_seq))
        encoded_seq = encoded_seq + [padding_id] * pad_length

    return encoded_sequence

def validate_seq_lens_within_margin(sequences: typing.List, max_len_diff: int) -> None:
    max_len = 0
    max_len_idx = -1
    min_len = 0
    min_len_idx = -1

    for i, seq in enumerate(sequences):
        if len(seq) > max_len:
            max_len = len(seq)
            max_len_idx = i
        if len(seq) < min_len:
            min_len = len(seq)
            min_len_idx = i

    if max_len - min_len > max_len_diff:
        raise ValueError(f"Sequence length difference for {sequences[max_len_idx]} and {sequences[min_len_idx]} ({max_len - min_len}) > {max_len_diff}")
    
    return

def interleave_csv_sequences(csv_sequences: typing.List[str]) -> typing.List[str]:
    split_sequences = [seq.split(',') for seq in csv_sequences]
    
    # Validate all lens are within one
    validate_seq_lens_within_margin(split_sequences, 1)

    zipped_tuples = zip_longest(*split_sequences, fillvalue=None)
    interleaved_sequence = [item for t in zipped_tuples for item in t if item is not None]

    return interleaved_sequence
 
def interleave_and_encode_csv_sequences(csv_sequences: typing.List[str], max_encoded_seq_len: int, id_mapping_table: typing.Dict[str, int], start_id: int, padding_id: int = 0) -> typing.List[int]:
    interleaved_sequence = interleave_csv_sequences(csv_sequences)
    
    try:
        encoded_sequence = encode_sequence(interleaved_sequence, max_encoded_seq_len, id_mapping_table, start_id, padding_id=padding_id)
    except Exception as e:
        raise ValueError(f"Failed to encode interleaved sequence {interleaved_sequence}: {e}")
    
    return encoded_sequence

class SequenceID:
    PITCHES = 'pitches'
    ARSENAL = 'arsenal'
    HANDEDNESS = 'handedness'
    ON_BASE = 'on_base'
    INTERLEAVED = 'interleaved'


@dataclass
class SequenceInfo:
    seq_id: SequenceID
    max_sequence_len: int
    vocab_ids: typing.List[vocab.VocabID]

@dataclass
class CSVSequenceInput:
    seq_id: SequenceID
    csv_sequence: str

ON_BASE_SEQ_INFO = SequenceInfo(SequenceID.ON_BASE, 3, [vocab.VocabID.BOOLEAN])


class PitchSequenceTokenizer:
    def __init__(
            self, 
            sequential_sequences: typing.List[SequenceInfo], 
            interleaved_sequence: SequenceInfo, 
            vocab_data: typing.List[vocab.VocabInfo] =[vocab.PITCH_VOCAB, vocab.COUNT_VOCAB, vocab.HANDEDNESS_VOCAB, vocab.BOOLEAN_VOCAB],
    ):
        self._padding_id = 0
        self._global_start_id = 1
        self.global_vocab = ['<pad>', '<start>']
        self.seq_start_token_ids = {}
        self.added_vocabs = set()
        self.max_sequence_lengths = {}

        # Add all of our core vocabs
        for v_data in vocab_data:
            if v_data.vocab_id in self.added_vocabs:
                raise ValueError(f"Duplicate vocab ids seen {v_data.vocab_id}")
            self.added_vocabs.add(v_data.vocab_id)
            self.global_vocab.extend(v_data.vocab)
        
        # Now add our sequence start tokens we might add.
        # Also check if our sequence's associated vocab is in our known vocab id set.
        current_free_idx = len(self.global_vocab)
        for seq in sequential_sequences:
            # Check if we know this vocab the sequence is trying to use.
            for v_id in seq.vocab_ids:
                if v_id not in self.added_vocabs:
                    raise ValueError(f"Vocab {v_id} not known to tokenizer for seq {seq.seq_id}")
                
            start_token_for_seq = f"<{seq.seq_id}_start>"
            self.global_vocab.append(start_token_for_seq)
            self.seq_start_token_ids[seq.seq_id] = current_free_idx
            # Add one for start token that will be added for each sequence.
            self.max_sequence_lengths[seq.seq_id] = seq.max_sequence_len + 1
            current_free_idx += 1
        
        # Check if we know this vocab the sequence is trying to use.
        for v_id in interleaved_sequence.vocab_ids:
            if v_id not in self.added_vocabs:
                raise ValueError(f"Vocab {v_id} not known to tokenizer for seq {interleaved_sequence.seq_id}")
        self.global_vocab.append("<interleaved_start>")
        self.seq_start_token_ids["interleaved"] = current_free_idx
        # Add one for start token that will be added for each sequence.
        self.max_sequence_lengths["interleaved"] = seq.max_sequence_len + 1
        current_free_idx += 1

        self.token_to_id = {}
        for i in range(0, len(self.global_vocab)):
            self.token_to_id[self.global_vocab[i]] = i


    def get_id_for_token(self, token: str) -> int:
        return self.token_to_id[token]
    
    def get_token_for_id(self, id: int) -> str:
        if id >= len(self.vocab_size()):
            raise ValueError(f"ID {id} not in vocab range < {len(self.self.vocab_size())}")
        
        return self.global_vocab[id]
    
    def vocab_size(self) -> int:
        return len(self.global_vocab)
    
    def tokenize(self, sequential_inputs: typing.List[CSVSequenceInput], csv_sequences_to_interleave: typing.List[str]) -> typing.Tuple[typing.List[int], typing.List[bool]]:
        final_encoded_sequence = [self._global_start_id]
        for input in sequential_inputs:
            if input.seq_id not in self.seq_start_token_ids:
                raise ValueError(f"Unknown sequence provided {input.seq_id}")
            
            seq_start_token_id = self.seq_start_token_ids[input.seq_id]
            max_len_for_seq = self.max_sequence_lengths[input.seq_id]
            try:
                encoded_sub_sequence = encode_sequence(input.csv_sequence.split(','), self.token_to_id, max_len_for_seq, seq_start_token_id, self._padding_id)
            except Exception as E:
                raise ValueError(f"Failed to encode sequential sequence input {input.seq_id}: {e}")
            
            final_encoded_sequence.extend(encoded_sub_sequence)
        
        # Now do interlaved sequences.
        interleaved_start_token_id = self.seq_start_token_ids['interleaved']
        max_len_for_interleaved_seq = self.max_sequence_lengths['interleaved']
        try:
            encoded_interleaved_sequence = interleave_and_encode_csv_sequences(csv_sequences_to_interleave, max_len_for_interleaved_seq, self.token_to_id, interleaved_start_token_id, self._padding_id)
        except Exception as e:
            raise ValueError(f"Failed to interleave and encode sequences {e}")

        final_encoded_sequence.extend(encoded_interleaved_sequence)
        # Don't need to check sizes as sizes are checks at component level.
        # Generate the padding mask for what the model should care about.
        padding_mask = [id == self._padding_id for id in final_encoded_sequence]

        return final_encoded_sequence, padding_mask

