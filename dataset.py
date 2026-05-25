import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tokenizer import Simpletokenizer

from config import (
    batch_size,
    max_seq_len,
    stride
)

class GPTDatasetV1(Dataset):

    def __init__(
        self,
        token_ids,
        max_length,
        stride
    ):

        self.input_ids = []

        self.target_ids = []

        for i in range(
            0,
            len(token_ids) - max_length,
            stride
        ):

            input_chunk = token_ids[
                i:i+max_length
            ]

            target_chunk = token_ids[
                i+1:i+max_length+1
            ]

            self.input_ids.append(
                torch.tensor(input_chunk)
            )

            self.target_ids.append(
                torch.tensor(target_chunk)
            )

    def __len__(self):

        return len(self.input_ids)

    def __getitem__(self, idx):

        return (
            self.input_ids[idx],
            self.target_ids[idx]
        )

def create_dataloader_v1(
    text,
    vocab
):

    tokenizer = Simpletokenizer(vocab)

    token_ids = tokenizer.encode(text)

    split_idx = int(
        0.9 * len(token_ids)
    )

    train_ids = token_ids[:split_idx]

    val_ids = token_ids[split_idx:]

    train_dataset = GPTDatasetV1(
        train_ids,
        max_seq_len,
        stride
    )

    val_dataset = GPTDatasetV1(
        val_ids,
        max_seq_len,
        stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader