import time
import numpy as np

from data_loader import make_batches
from models import ElmanRNN, JordanRNN


def train(
    elman,
    jordan,
    dataset,
    seq_length,
    num_epochs,
):
    train_data = dataset["train_data"]
    val_data   = dataset["val_data"]
    vocab_size = dataset["vocab_size"]
    hidden_size = elman.hidden_size

    elman_h0  = np.zeros((hidden_size, 1))
    jordan_h0 = np.zeros((hidden_size, 1))
    jordan_p0 = np.zeros((vocab_size, 1))

    history = _empty_history()

    _print_train_header(elman, jordan, dataset, seq_length, num_epochs)

    batches_per_epoch = sum(1 for _ in make_batches(train_data, seq_length))
    print(f"  Training batches per epoch : {batches_per_epoch:,}")
    print(f"  Validation size            : {len(val_data):,} chars")
    print("=" * 72)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        elman_h  = elman_h0.copy()
        jordan_h = jordan_h0.copy()
        jordan_p = jordan_p0.copy()

        elman_batch_losses  = []
        jordan_batch_losses = []
        elman_correct = jordan_correct = 0
        total_chars_trained = 0

        print(f"\n{'-'*72}")
        print(f"  Epoch {epoch:>3d}/{num_epochs}  |  Training...")
        print(f"{'-'*72}")

        log_interval = max(1, batches_per_epoch // 5)

        for batch_num, (inputs, targets) in enumerate(make_batches(train_data, seq_length)):
            elman_loss, elman_h, elman_preds = elman.train_step(inputs, targets, elman_h)
            jordan_loss, jordan_h, jordan_p, jordan_preds = jordan.train_step(inputs, targets, jordan_h, jordan_p)

            elman_batch_losses.append(elman_loss)
            jordan_batch_losses.append(jordan_loss)

            elman_correct  += sum(p == t for p, t in zip(elman_preds, targets))
            jordan_correct += sum(p == t for p, t in zip(jordan_preds, targets))
            total_chars_trained += len(targets)

            if (batch_num + 1) % log_interval == 0 or batch_num == batches_per_epoch - 1:
                pct = 100.0 * (batch_num + 1) / batches_per_epoch
                print(
                    f"    [{pct:5.1f}%] batch {batch_num+1:>5}/{batches_per_epoch}"
                    f"  |  Elman loss {elman_loss:.4f}"
                    f"  |  Jordan loss {jordan_loss:.4f}"
                )

        elman_train_loss = float(np.mean(elman_batch_losses))
        jordan_train_loss = float(np.mean(jordan_batch_losses))
        elman_train_acc  = elman_correct / total_chars_trained
        jordan_train_acc = jordan_correct / total_chars_trained
        elman_train_ppl  = float(np.exp(elman_train_loss))
        jordan_train_ppl = float(np.exp(jordan_train_loss))

        elman_val_loss,  elman_val_acc,  elman_val_ppl  = _validate_elman(elman,   val_data, seq_length, hidden_size)
        jordan_val_loss, jordan_val_acc, jordan_val_ppl = _validate_jordan(jordan, val_data, seq_length, hidden_size, vocab_size)

        epoch_time = time.time() - epoch_start
        history["epoch_times"].append(epoch_time)

        for model_name, split, loss, acc, ppl in [
            ("elman",  "train", elman_train_loss,  elman_train_acc,  elman_train_ppl),
            ("elman",  "val",   elman_val_loss,    elman_val_acc,    elman_val_ppl),
            ("jordan", "train", jordan_train_loss, jordan_train_acc, jordan_train_ppl),
            ("jordan", "val",   jordan_val_loss,   jordan_val_acc,   jordan_val_ppl),
        ]:
            history[f"{model_name}_{split}_loss"].append(loss)
            history[f"{model_name}_{split}_acc"].append(acc)
            history[f"{model_name}_{split}_ppl"].append(ppl)

        _print_epoch_summary(
            epoch, epoch_time,
            elman_train_loss,  elman_train_acc,  elman_train_ppl,
            elman_val_loss,    elman_val_acc,    elman_val_ppl,
            jordan_train_loss, jordan_train_acc, jordan_train_ppl,
            jordan_val_loss,   jordan_val_acc,   jordan_val_ppl,
        )

    print("\n" + "=" * 72)
    print("  Training complete.")
    print("=" * 72 + "\n")

    return history


def _validate_elman(model, val_data, seq_length, hidden_size):
    hidden = np.zeros((hidden_size, 1))
    losses, correct, total = [], 0, 0

    for inputs, targets in make_batches(val_data, seq_length):
        loss, hidden, preds = model.evaluate_step(inputs, targets, hidden)
        losses.append(loss)
        correct += sum(p == t for p, t in zip(preds, targets))
        total   += len(targets)

    if not losses:
        return 0.0, 0.0, 1.0

    mean_loss = float(np.mean(losses))
    return mean_loss, correct / total, float(np.exp(mean_loss))


def _validate_jordan(model, val_data, seq_length, hidden_size, vocab_size):
    hidden    = np.zeros((hidden_size, 1))
    out_probs = np.zeros((vocab_size, 1))
    losses, correct, total = [], 0, 0

    for inputs, targets in make_batches(val_data, seq_length):
        loss, hidden, out_probs, preds = model.evaluate_step(inputs, targets, hidden, out_probs)
        losses.append(loss)
        correct += sum(p == t for p, t in zip(preds, targets))
        total   += len(targets)

    if not losses:
        return 0.0, 0.0, 1.0

    mean_loss = float(np.mean(losses))
    return mean_loss, correct / total, float(np.exp(mean_loss))


def _empty_history():
    keys = [
        "elman_train_loss",  "elman_val_loss",
        "elman_train_acc",   "elman_val_acc",
        "elman_train_ppl",   "elman_val_ppl",
        "jordan_train_loss", "jordan_val_loss",
        "jordan_train_acc",  "jordan_val_acc",
        "jordan_train_ppl",  "jordan_val_ppl",
        "epoch_times",
    ]
    return {k: [] for k in keys}


def _print_train_header(elman, jordan, dataset, seq_length, num_epochs):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  TINY SHAKESPEARE MEMORY DUEL - Training Session")
    print(sep)
    print(
        f"  Dataset      : {dataset['used_chars']:,} / {dataset['total_chars']:,} chars "
        f"({dataset['data_fraction']*100:.0f}% of full corpus)"
    )
    print(f"  Vocabulary   : {dataset['vocab_size']} unique characters")
    print(f"  Train / Val  : {dataset['num_train']:,} / {dataset['num_val']:,} chars")
    print(f"  Seq length   : {seq_length}")
    print(f"  Epochs       : {num_epochs}")
    print(f"\n  {elman}")
    print(f"  {jordan}")


def _print_epoch_summary(
    epoch, epoch_time,
    e_tr_loss, e_tr_acc, e_tr_ppl,
    e_val_loss, e_val_acc, e_val_ppl,
    j_tr_loss, j_tr_acc, j_tr_ppl,
    j_val_loss, j_val_acc, j_val_ppl,
):
    w   = 72
    sep = "-" * w
    print(f"\n  {sep}")
    print(f"  |  EPOCH {epoch:>3d} SUMMARY  ({epoch_time:.1f}s){' '*(w-32)}|")
    print(f"  |-{sep}-|")
    print(f"  |  {'Metric':<22} {'Elman Train':>12} {'Elman Val':>11} {'Jordan Train':>13} {'Jordan Val':>11}  |")
    print(f"  |-{sep}-|")
    print(f"  |  {'CE Loss':<22} {e_tr_loss:>12.4f} {e_val_loss:>11.4f} {j_tr_loss:>13.4f} {j_val_loss:>11.4f}  |")
    print(f"  |  {'Accuracy (%)':<22} {e_tr_acc*100:>11.2f}% {e_val_acc*100:>10.2f}% {j_tr_acc*100:>12.2f}% {j_val_acc*100:>10.2f}%  |")
    print(f"  |  {'Perplexity':<22} {e_tr_ppl:>12.2f} {e_val_ppl:>11.2f} {j_tr_ppl:>13.2f} {j_val_ppl:>11.2f}  |")
    print(f"  {sep}")
