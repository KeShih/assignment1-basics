from cs336_basics.bpe_trainer import train_bpe
import json

path = "data/owt_train.txt"
N = 32000

if __name__ == "__main__":
    vocab, merges = train_bpe(path, N, special_tokens=["<|endoftext|>"])

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}, f, ensure_ascii=False, indent=2)

    with open("merges.json", "w", encoding="utf-8") as f:
        json.dump([(a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace")) for a, b in merges], f, ensure_ascii=False, indent=2)
