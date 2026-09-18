"""BPE tokenizer training.

Trains a byte-level BPE tokenizer from a text corpus, returning the
vocabulary and the ordered list of merges.
"""

from __future__ import annotations
from multiprocessing import Pool
import regex as re
from functools import partial

from cs336_basics.pretokenization_example import find_chunk_boundaries

import os

import heapq

import regex as re
import os

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # init
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # special tokens
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # chunk
    n_chunk = 128
    bdr = find_chunk_boundaries(open(input_path, "rb"), n_chunk, b"<|endoftext|>")
    start_end_pairs = list(zip(bdr[:-1], bdr[1:]))
    
    n_processes = 8
    with Pool(processes=n_processes) as pool:
        counters = pool.map(
            partial(process_chunk, input_path=input_path, special_tokens=special_tokens),
            start_end_pairs)
        
    # merge counters
    counter = {}
    for c in counters:
        for word, freq in c.items():
            counter[word] = counter.get(word, 0) + freq
            
    # BPE training loop
    
    pair_counter = {} # pair -> frequency
    pair_word_map = {} # pair -> words
    for word, freq in counter.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counter[pair] = pair_counter.get(pair, 0) + freq
            if pair not in pair_word_map:
                pair_word_map[pair] = set()
            pair_word_map[pair].add(word)
    
    while len(vocab) < vocab_size:
        print(f"vocab size: {len(vocab)}")
        if not pair_counter:
            break
        
        # merge
        best_pair = max(pair_counter, key=lambda pair: (pair_counter[pair],pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        
        # update counter
        for word in list(pair_word_map[best_pair]):
            # x,a,b,y -> x,ab,y
            # (x,a), (a,b), (b,y) -> (x,ab), (ab,y)
            seen_pairs = {}
            freq = counter[word]
            
            # build new word
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            
            # count pairs in old word
            old_pair_counts = {}
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                old_pair_counts[pair] = old_pair_counts.get(pair, 0) + freq
            
            # count pairs in new word
            new_pair_counts = {}
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                new_pair_counts[pair] = new_pair_counts.get(pair, 0) + freq
                
            pairs_to_update = set(old_pair_counts.keys()).union(set(new_pair_counts.keys()))
            for pair in pairs_to_update:
                old_count = old_pair_counts.get(pair, 0)
                new_count = new_pair_counts.get(pair, 0)
                pair_counter[pair] = pair_counter.get(pair, 0) - old_count + new_count
                # remove old word
                if old_count > 0:
                    pair_word_map[pair].discard(word)
                # add new_word if new_count > 0 
                if new_count > 0:
                    if pair not in pair_word_map:
                        pair_word_map[pair] = set()
                    pair_word_map[pair].add(new_word)
                
            counter[new_word] = counter.get(new_word, 0) + freq
            del counter[word]   
        
        del pair_counter[best_pair]
        del pair_word_map[best_pair]
        
        for pair in list(pair_counter.keys()):
            if pair_counter[pair] <= 0:
                del pair_counter[pair]
                del pair_word_map[pair]
        
    return vocab, merges

def process_chunk(start_end_pair, input_path, special_tokens):
        start, end = start_end_pair[0], start_end_pair[1]
        print(f"Processing chunk: {start} - {end}")
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            return pretokenize(chunk, special_tokens)

def pretokenize(text: str, special_tokens: list[str]) -> list[list[bytes]]:
    # split on special tokens
    if special_tokens:
        pattern = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    # pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    counter = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            token = match.group(0)
            # tuple can be used as a key in a dict, while list cannot
            word = tuple(
                bytes([b])
                for b in token.encode("utf-8")
            )
            counter[word] = counter.get(word, 0) + 1
            
    return counter



