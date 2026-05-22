import numpy as np
from models import ElmanRNN, JordanRNN
from data_loader import decode


def generate_elman(
    model,
    seed_text,
    char2idx,
    idx2char,
    length,
    temperature = 1.0,
):
    seed_idx = _encode_seed(seed_text, char2idx)
    h_init   = np.zeros((model.hidden_size, 1))
    output_idx = model.generate(
        seed_indices=seed_idx,
        h_init=h_init,
        length=length,
        temperature=temperature,
    )
    return decode(output_idx, idx2char)


def generate_jordan(
    model,
    seed_text,
    char2idx,
    idx2char,
    length,
    temperature = 1.0,
):
    seed_idx = _encode_seed(seed_text, char2idx)
    h_init   = np.zeros((model.hidden_size, 1))
    p_init   = np.zeros((model.vocab_size, 1))
    output_idx = model.generate(
        seed_indices=seed_idx,
        h_init=h_init,
        p_init=p_init,
        length=length,
        temperature=temperature,
    )
    return decode(output_idx, idx2char)


def temperature_experiment(
    elman,
    jordan,
    seed_text,
    char2idx,
    idx2char,
    temperatures,
    gen_length,
):
    return {
        temp: {
            "elman":  generate_elman(elman,   seed_text, char2idx, idx2char, gen_length, temp),
            "jordan": generate_jordan(jordan, seed_text, char2idx, idx2char, gen_length, temp),
        }
        for temp in temperatures
    }


def print_generation_report(seed_text, results, save_path = None):
    lines = []
    bar   = "=" * 72

    lines.append(bar)
    lines.append("  TEXT GENERATION REPORT")
    lines.append(bar)
    lines.append(f"  Seed text : {repr(seed_text)}")
    lines.append("")

    for temp, texts in sorted(results.items()):
        lines.append(f"{'-'*72}")
        lines.append(f"  Temperature t = {temp}")
        lines.append(f"{'-'*72}")
        lines.append("\n  [ ELMAN - output ]")
        lines.append("  " + texts["elman"].replace("\n", "\n  "))
        lines.append("\n  [ JORDAN - output ]")
        lines.append("  " + texts["jordan"].replace("\n", "\n  "))
        lines.append("")

    lines.append(bar)
    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  [Done] Generation report saved -> {save_path}")


def _encode_seed(seed_text, char2idx):
    indices = []
    for ch in seed_text:
        if ch in char2idx:
            indices.append(char2idx[ch])
        else:
            print(f"  [WARN] Seed character {repr(ch)} not in vocabulary - skipped.")
    if not indices:
        raise ValueError("Seed text produced no valid indices. Check your vocabulary.")
    return indices
