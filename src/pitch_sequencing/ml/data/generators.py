import abc
import math
import pandas as pd

from typing import Optional, Type, Any

from pitch_sequencing.ml.tokenizers.pitch_arsenal import PitchArsenalLookupTable

# TODO(kaelen): probably consider a different name besides Generator since it's used in python
class CSVSequenceGenerator(abc.ABC):
    @abc.abstractmethod
    def generate_csv_sequence_from_df_row(self, row: pd.Series) -> str:
        pass

class DirectCSVLookupGenerator(CSVSequenceGenerator):
    def __init__(self, df_key: str, type_conversion: Optional[Type[Any]] = None):
        self.df_key = df_key
        self.type_conversion = type_conversion

    def generate_csv_sequence_from_df_row(self, row: pd.Series) -> str:
        value = row[self.df_key]
        if self.type_conversion is not None:
            value = value.astype(self.type_conversion)
        return str(value)
    
class OnBaseCSVGenerator(CSVSequenceGenerator):
    def __init__(self, first_base_df_key: str = 'on_1b', second_base_df_key: str = 'on_2b', third_base_df_key: str = 'on_3b'):
        self.first_base_df_key = first_base_df_key
        self.second_base_df_key = second_base_df_key
        self.third_base_df_key = third_base_df_key
    
    def generate_csv_sequence_from_df_row(self, row: pd.Series) -> str:
        batter_on_first = 'F' if math.isnan(row[self.first_base_df_key]) else 'T'
        batter_on_second = 'F' if math.isnan(row[self.second_base_df_key]) else 'T'
        batter_on_third = 'F' if math.isnan(row[self.third_base_df_key]) else 'T'

        return ",".join([batter_on_first, batter_on_second, batter_on_third])

class HandednessCSVGenerator(CSVSequenceGenerator):
    def __init__(self, pitcher_df_key: str = 'p_throws', batter_df_key: str = 'stand'):
        self.pitcher_df_key = pitcher_df_key
        self.batter_df_key = batter_df_key

    def generate_csv_sequence_from_df_row(self, row: pd.Series) -> str:
        return f"{row[self.pitcher_df_key]},{row[self.batter_df_key]}"

class ArsenalCSVGenerator(CSVSequenceGenerator):
    def __init__(self, arsenal_lookup_table: PitchArsenalLookupTable, pitcher_id_df_key: str = 'pitcher_id'):
        self.lookup_table = arsenal_lookup_table
        self.pitcher_id_df_key = pitcher_id_df_key

    def generate_csv_sequence_from_df_row(self, row: pd.Series) -> str:
        pitcher_id = row[self.pitcher_id_df_key]

        try:
            arsenal = self.lookup_table.arsenal_for_pitcher_id(pitcher_id)
        except Exception as e:
            raise ValueError(f"Failed to get pitcher arsenal for {pitcher_id}: {e}")

        return arsenal
    