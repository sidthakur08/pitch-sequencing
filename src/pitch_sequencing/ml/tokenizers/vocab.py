import typing

from dataclasses import dataclass

# Generated using
# gs://pitch-sequencing/sequence_data/large_sequence_data_cur_opt.csv
ORDERED_PITCHES=['CB', 'KN', 'FC', 'FS', 'CH', 'FF', 'SL', 'PO', 'SI', 'ST']
ORDERED_COUNT=['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']
HANDEDNESS=['L', 'R']
BOOLEAN = ['T', 'F']

class VocabID:
    PITCHES = 'pitches',
    COUNTS = 'counts',
    HANDEDNESS = 'handedness'
    BOOLEAN = 'boolean'

@dataclass
class VocabInfo:
    vocab_id: VocabID
    vocab: typing.List[str]

VOCAB_INFO_LOOKUP = {
    VocabID.PITCHES: ORDERED_PITCHES,
    VocabID.COUNTS: ORDERED_COUNT,
    VocabID.HANDEDNESS: HANDEDNESS,
    VocabID.BOOLEAN: BOOLEAN,
}

PITCH_VOCAB = VocabInfo(VocabID.PITCHES, ORDERED_PITCHES)
COUNT_VOCAB = VocabInfo(VocabID.COUNTS, ORDERED_COUNT)
HANDEDNESS_VOCAB = VocabInfo(VocabID.HANDEDNESS, HANDEDNESS)
BOOLEAN_VOCAB = VocabInfo(VocabID.BOOLEAN, BOOLEAN)
