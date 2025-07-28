import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tinywhisper.config.config import Config
from tinywhisper.data.dataset import HFDataset

config = Config()


def custom_collate_fn(batch, pad_token_id: int = config.training.pad_token_id):
    input_feats = [item["input_features"] for item in batch]  # (T, n_mels)
    labels = [item["labels"] for item in batch]

    # original lengths
    input_lens = torch.tensor([x.shape[0] for x in input_feats], dtype=torch.long)
    label_lens = torch.tensor([x.shape[0] for x in labels], dtype=torch.long)

    # padding input features (along time) - 0.0 default
    padded_feats = torch.nn.utils.rnn.pad_sequence(
        input_feats, batch_first=True
    )  # (B, T_max, n_mels)
    # padding labels with pad token ID
    # Pad labels manually using pad_token_id
    max_label_len = label_lens.max().item()
    padded_labels = []
    for label in labels:
        padding_len = max_label_len - label.shape[0]
        padded_label = F.pad(label, (0, padding_len), value=pad_token_id)
        padded_labels.append(padded_label)

    padded_labels = torch.stack(padded_labels, dim=0)  # (B, L_max)

    return {
        "input_features": padded_feats,  # → (B, T_max, n_mels)
        "input_lengths": input_lens,  # (B,)
        "labels": padded_labels,  # (B, L_max)
        "label_lengths": label_lens,  # (B,)
    }


def get_dataloader(
    config: Config,
    batch_size: int = 16,
) -> dict:
    dataset = HFDataset(dataset_name=config.dataset.hf_dataset_name)
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    train_val_split = train_dataset.train_test_split(
        test_size=0.15
    )  # 15% of the training set for validation
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }


def get_all_dataloaders(config):
    return {
        "train": get_dataloader(config, "train"),
        "val": get_dataloader(config, "val"),
        "test": get_dataloader(config, "test"),
    }
