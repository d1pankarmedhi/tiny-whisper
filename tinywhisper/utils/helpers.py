import torch
from jiwer import wer
from tqdm import tqdm

from tinywhisper.model.model import TinyWhisper
from tinywhisper.tokenizer import BPETokenizer


def greedy_decoder(
    model: TinyWhisper,
    input_feats,
    max_len=100,
    sos_token_id=50257,
    eos_token_id=50258,
):
    model.eval()
    batch_size = input_feats.size(0)
    device = input_feats.device

    # Start with SOS token for all samples
    decoder_input = torch.full((batch_size, 1), sos_token_id, dtype=torch.long).to(
        device
    )

    for _ in range(max_len):
        # Build causal mask
        T = decoder_input.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        tgt_mask = tgt_mask.masked_fill(tgt_mask, float("-inf"))

        with torch.no_grad():
            logits = model(
                input_feats, decoder_input, target_mask=tgt_mask
            )  # (B, T, V)

        next_token_logits = logits[:, -1, :]  # (B, V) - only last step
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)

        decoder_input = torch.cat([decoder_input, next_tokens], dim=1)

    return decoder_input[:, 1:]  # Remove SOS


def evaluate_autoregressive(
    model: TinyWhisper,
    dataloader,
    tokenizer: BPETokenizer,
    criterion,
    device="cuda",
    sos_token_id=50257,
    eos_token_id=50258,
    pad_token_id=50259,
    n_examples=5,
):
    model.eval()
    predictions = []
    references = []

    shown = 0
    printed_examples = []

    total_val_loss = 0
    total_tokens = 0
    special_tokens = [sos_token_id, eos_token_id, pad_token_id]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
                tokenizer,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id,
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
            if shown < n_examples:
                for ref, pred in zip(label_texts, pred_texts):
                    if shown >= n_examples:
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

    return avg_val_loss, val_wer
