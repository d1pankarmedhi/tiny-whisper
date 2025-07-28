import os

import torch
import torch.nn as nn
import torch.optim as optim
from config.config import Config
from data.dataset import SpeechDataset, custom_collate_fn
from jiwer import wer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tinywhisper.model.model import TinyWhisper
from tinywhisper.tokenizer import BPETokenizer
from tinywhisper.utils.helpers import greedy_decoder


def train():
    # Load config
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup paths
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)

    # Dataset & DataLoader
    train_set = SpeechDataset(config.paths.train_manifest, config)
    val_set = SpeechDataset(config.paths.val_manifest, config)

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # Model & Tokenizer
    model = TinyWhisper(config).to(device)
    tokenizer = BPETokenizer()

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=0.01
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        total_steps=config.training.num_epochs
        * len(train_loader)
        // config.training.accumulation_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
    scaler = torch.amp.GradScaler()

    best_val_wer = float("inf")

    # Training Loop
    for epoch in range(config.training.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for i, batch in enumerate(pbar):
            input_feats = batch["input_features"].to(device)
            input_lens = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)

            # Target input is all tokens except the last
            # Target output is all tokens except the first
            decoder_input = labels[:, :-1]
            target_output = labels[:, 1:]

            # Causal mask
            T = decoder_input.size(1)
            tgt_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)
            tgt_mask = tgt_mask.masked_fill(tgt_mask, float("-inf"))

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    input_feats, input_lens, decoder_input, target_mask=tgt_mask
                )  # (B, T, V)
                logits = logits.reshape(-1, logits.shape[-1])  # (B*T, V)
                target_output = target_output.reshape(-1)  # (B*T,)

                loss = (
                    criterion(logits, target_output)
                    / config.training.accumulation_steps
                )

            scaler.scale(loss).backward()
            total_loss += loss.item() * config.training.accumulation_steps

            if (i + 1) % config.training.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix(loss=total_loss / (i + 1))

        print(f"Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader):.4f}")

        # Evaluation
        model.eval()
        predictions = []
        references = []

        shown = 0
        printed_examples = []

        total_val_loss = 0
        total_tokens = 0
        special_tokens = [
            config.training.sos_token_id,
            config.training.eos_token_id,
            config.training.pad_token_id,
        ]

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_feats = batch["input_features"].to(device)
                input_lens = batch["input_lengths"].to(device)
                label_ids = batch["labels"].to(device)

                # Prepare decoder input/target
                decoder_input = label_ids[:, :-1]
                target_output = label_ids[:, 1:]

                T = decoder_input.size(1)
                tgt_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)
                tgt_mask = tgt_mask.masked_fill(tgt_mask, float("-inf"))

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(
                        input_feats, input_lens, decoder_input, target_mask=tgt_mask
                    )
                    logits_flat = logits.reshape(-1, logits.shape[-1])
                    target_output_flat = target_output.reshape(-1)

                    loss = criterion(logits_flat, target_output_flat)
                    total_val_loss += loss.item()
                    total_tokens += 1

                # Decode predictions
                pred_ids = greedy_decoder(
                    model,
                    input_feats,
                    sos_token_id=config.training.sos_token_id,
                    eos_token_id=config.training.eos_token_id,
                )

                # Convert to string, excluding special tokens
                pred_texts = [
                    tokenizer.decode(
                        [id for id in ids.tolist() if id not in special_tokens]
                    )
                    for ids in pred_ids
                ]
                # Convert to string, excluding special tokens, iterating over list of ids
                label_texts = [
                    tokenizer.decode(
                        [id for id in label.tolist() if id not in special_tokens]
                    )
                    for label in label_ids.cpu()
                ]

                predictions.extend(pred_texts)
                references.extend(label_texts)

                # Show examples
                if shown < 3:
                    for ref, pred in zip(label_texts, pred_texts):
                        if shown >= 3:
                            break
                        printed_examples.append((ref, pred))
                        shown += 1

        avg_val_loss = total_val_loss / total_tokens
        val_wer = wer(references, predictions)

        # Print examples
        print("\n📝 Sample Predictions vs References:")
        for i, (ref, pred) in enumerate(printed_examples):
            print(f"\nExample {i + 1}")
            print(f"Ground Truth: {ref}")
            print(f"Prediction  : {pred}")

        print(f"\n📉 Validation Loss: {avg_val_loss:.4f}")
        print(f"🗣️  Validation WER : {val_wer:.4f}")

        # Save best model
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            torch.save(model.state_dict(), config.paths.best_model_path)
            print(f"✅ Saved new best model with WER: {val_wer:.4f}")


if __name__ == "__main__":
    train()
