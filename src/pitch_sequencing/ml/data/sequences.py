import typing

import torch

from dataclasses import dataclass

@dataclass
class SingularSequence:
    """
    src: tokenized sequence of type long. Size 1xN
    src_mask: boolean tensor indicating padding of src. Size 1xN.
    """
    src: torch.Tensor
    src_mask: torch.Tensor

    def to(self, device: torch.device) -> 'SingularSequence':
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)

        return self

    def unsqueeze(self, dim: int) -> 'SingularSequence':
        self.src = self.src.unsqueeze(dim)
        self.src_mask = self.src_mask.unsqueeze(dim)

        return self


# TODO(kaelen) figure out how to not write these for each type
def collate_interleaved_and_target(batch) -> typing.Tuple[SingularSequence, torch.Tensor]:
    seq_data_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])

    srcs = torch.stack([data.src for data in seq_data_list])
    src_masks = torch.stack([data.src_mask for data in seq_data_list])

    return SingularSequence(srcs, src_masks), targets
