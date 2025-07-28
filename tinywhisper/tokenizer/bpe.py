import tiktoken


class BPETokenizer:
    def __init__(
        self,
        special_tokens=["<|startoftext|>", "<|endoftext|>", "<|pad|>"],
        model_name="gpt2",
    ):
        self.model_name = model_name
        self.special_tokens = special_tokens
        self.tokenizer = self.extend_tokenizer()

    def encode(self, text: str):
        return self.tokenizer.encode(
            text, allowed_special={"<|pad|>", "<|startoftext|>", "<|endoftext|>"}
        )

    def decode(self, tokens: list):
        return self.tokenizer.decode(tokens)

    @property
    def n_vocab(self):
        return self.tokenizer.n_vocab  # 50260

    # add special tokens to vocabulary
    def extend_tokenizer(self):
        tokenizer = tiktoken.get_encoding(self.model_name)
        special_token_ids = {
            token: tokenizer.n_vocab + i for i, token in enumerate(self.special_tokens)
        }
        extended_tokenizer = tiktoken.Encoding(
            name="gpt2_extended",
            pat_str=tokenizer._pat_str,
            mergeable_ranks=tokenizer._mergeable_ranks,
            special_tokens={
                **tokenizer._special_tokens,
                **special_token_ids,
            },
        )
        return extended_tokenizer
