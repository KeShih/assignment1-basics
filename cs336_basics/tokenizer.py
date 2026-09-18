"""Byte-level BPE tokenizer."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        self.decoder = {v: k for k, v in vocab.items()}
        self.special_tokens_set = set(special_tokens) if special_tokens else set()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = {int(k): v.encode("utf-8") for k, v in json.load(f).items()}
            
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [
                (a.encode("utf-8"), b.encode("utf-8"))
                for a, b in json.load(f)
            ]
        return cls(vocab, merges, special_tokens)


    def _apply_bpe(self, token: str) -> list[int]:
        # Apply BPE to a single token and return the list of token IDs.
        bs = [bytes([b]) for b in token.encode("utf-8")]

        while True:
            best_pair = None
            best_rank = float("inf")

            # Find the applicable pair with smallest merge rank.
            for i in range(len(bs) - 1):
                pair = (bs[i], bs[i + 1])
                if pair in self.merge_rank and self.merge_rank[pair] < best_rank:
                    best_pair = pair
                    best_rank = self.merge_rank[pair]

            if best_pair is None:
                break

            # Merge via the best pair
            new_bs = []
            i = 0
            while i < len(bs):
                if i < len(bs)-1 and (bs[i], bs[i + 1]) == best_pair:
                    new_bs.append(bs[i] + bs[i + 1])
                    i += 2
                else:
                    new_bs.append(bs[i])
                    i += 1
            bs = new_bs
        return [self.decoder[b] for b in bs]

    # divide text via special tokens, then devide via GPT-2 pre-tokenization regex
    def _pretokenize(self, text: str) -> Iterator[str]:
        # split on special tokens
        if self.special_tokens:
            pattern = "(" + "|".join(
                re.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        # pre-tokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        for chunk in chunks:
            if chunk in self.special_tokens_set:
                yield chunk
            else:
                for match in re.finditer(PAT, chunk):
                    token = match.group(0)
                    yield token
    

    def encode(self, text: str) -> Iterator[int]:
        ids = []
        for token in self._pretokenize(text):
            if token in self.special_tokens_set:
                ids.append(self.decoder[token.encode("utf-8")])
            else:
                ids.extend(self._apply_bpe(token))
        return ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
    
    