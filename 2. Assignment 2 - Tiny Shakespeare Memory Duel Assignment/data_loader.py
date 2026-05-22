import os
import zipfile
import numpy as np


def load_and_prepare(
    zip_path,
    data_fraction=0.25,
    train_ratio=0.90,
    seed=42,
):
    np.random.seed(seed)

    full_text = _read_zip(zip_path)
    total_chars = len(full_text)
    num_chars_used = max(1000, int(total_chars * data_fraction))
    text = full_text[:num_chars_used]

    char2idx, idx2char = _build_vocab(text)
    encoded = _encode(text, char2idx)

    train_split_idx = int(len(encoded) * train_ratio)
    train_data = encoded[:train_split_idx]
    val_data = encoded[train_split_idx:]

    return dict(
        train_data=train_data,
        val_data=val_data,
        char2idx=char2idx,
        idx2char=idx2char,
        vocab_size=len(char2idx),
        text_sample=text[:200],
        num_train=len(train_data),
        num_val=len(val_data),
        data_fraction=data_fraction,
        total_chars=total_chars,
        used_chars=num_chars_used,
    )


def decode(indices, idx2char):
    return "".join(idx2char[int(i)] for i in indices)


def make_batches(data, seq_length):
    n = len(data)
    for i in range(0, n - seq_length, seq_length):
        inputs = data[i: i + seq_length].tolist()
        targets = data[i + 1: i + seq_length + 1].tolist()
        yield inputs, targets


def _read_zip(zip_path):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Dataset zip not found at '{zip_path}'.\n"
            "Please place tiny-shakespeare.zip in the data/ directory."
        )
    with zipfile.ZipFile(zip_path, 'r') as zf:
        txt_files = [n for n in zf.namelist() if n.endswith('.txt')]
        if not txt_files:
            raise ValueError("No .txt file found inside the zip archive.")
        return zf.read(txt_files[0]).decode('utf-8')


def _build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char


def _encode(text, char2idx):
    return np.array([char2idx[c] for c in text], dtype=np.int32)
