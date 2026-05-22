import io
import math
import numpy as np


def cross_entropy(loss_per_char):
    return float(loss_per_char)


def perplexity(loss_per_char):
    return float(math.exp(min(loss_per_char, 700)))


def accuracy(correct_count, total_count):
    if total_count == 0:
        return 0.0
    return float(correct_count) / float(total_count)


def moving_average(values, window = 5):
    result = list(values)
    for i in range(window - 1, len(values)):
        result[i] = float(np.mean(values[max(0, i - window + 1): i + 1]))
    return result


def compute_final_metrics(history):
    def best_val_epoch(losses):
        return int(np.argmin(losses)) + 1

    e_best_ep = best_val_epoch(history["elman_val_loss"])
    j_best_ep = best_val_epoch(history["jordan_val_loss"])

    return {
        "elman": {
            "train_loss_final": history["elman_train_loss"][-1],
            "val_loss_final":   history["elman_val_loss"][-1],
            "train_acc_final":  history["elman_train_acc"][-1],
            "val_acc_final":    history["elman_val_acc"][-1],
            "train_ppl_final":  history["elman_train_ppl"][-1],
            "val_ppl_final":    history["elman_val_ppl"][-1],
            "best_val_epoch":   e_best_ep,
            "best_val_loss":    history["elman_val_loss"][e_best_ep - 1],
            "best_val_acc":     history["elman_val_acc"][e_best_ep - 1],
            "best_val_ppl":     history["elman_val_ppl"][e_best_ep - 1],
        },
        "jordan": {
            "train_loss_final": history["jordan_train_loss"][-1],
            "val_loss_final":   history["jordan_val_loss"][-1],
            "train_acc_final":  history["jordan_train_acc"][-1],
            "val_acc_final":    history["jordan_val_acc"][-1],
            "train_ppl_final":  history["jordan_train_ppl"][-1],
            "val_ppl_final":    history["jordan_val_ppl"][-1],
            "best_val_epoch":   j_best_ep,
            "best_val_loss":    history["jordan_val_loss"][j_best_ep - 1],
            "best_val_acc":     history["jordan_val_acc"][j_best_ep - 1],
            "best_val_ppl":     history["jordan_val_ppl"][j_best_ep - 1],
        },
    }


def print_final_metrics_report(final, total_epochs):
    e   = final["elman"]
    j   = final["jordan"]
    bar = "=" * 72
    col = 22

    print(f"\n{bar}")
    print("  FINAL EVALUATION METRICS SUMMARY")
    print(bar)
    print(f"  {'Metric':<{col}}{'Elman':>18}{'Jordan':>18}")
    print(f"  {'-'*56}")

    rows = [
        ("Training Loss (CE)",    f"{e['train_loss_final']:.4f}",    f"{j['train_loss_final']:.4f}"),
        ("Validation Loss (CE)",  f"{e['val_loss_final']:.4f}",      f"{j['val_loss_final']:.4f}"),
        ("Training Accuracy (%)", f"{e['train_acc_final']*100:.2f}", f"{j['train_acc_final']*100:.2f}"),
        ("Validation Accuracy (%)", f"{e['val_acc_final']*100:.2f}", f"{j['val_acc_final']*100:.2f}"),
        ("Training Perplexity",   f"{e['train_ppl_final']:.2f}",     f"{j['train_ppl_final']:.2f}"),
        ("Validation Perplexity", f"{e['val_ppl_final']:.2f}",       f"{j['val_ppl_final']:.2f}"),
    ]
    for label, e_val, j_val in rows:
        print(f"  {label:<{col}}{e_val:>18}{j_val:>18}")

    print(f"\n  {'-'*56}")
    print(f"  {'Best Validation Epoch':<{col}}{e['best_val_epoch']:>18}{j['best_val_epoch']:>18}")
    print(f"  {'Best Val Loss (CE)':<{col}}{e['best_val_loss']:>18.4f}{j['best_val_loss']:>18.4f}")
    print(f"  {'Best Val Accuracy (%)':<{col}}{e['best_val_acc']*100:>17.2f}%{j['best_val_acc']*100:>17.2f}%")
    print(f"  {'Best Val Perplexity':<{col}}{e['best_val_ppl']:>18.2f}{j['best_val_ppl']:>18.2f}")
    print(f"\n  Total epochs trained : {total_epochs}")
    print(bar)

    elman_wins = sum([
        e["val_loss_final"] < j["val_loss_final"],
        e["val_acc_final"]  > j["val_acc_final"],
        e["val_ppl_final"]  < j["val_ppl_final"],
    ])
    print(f"\n  Elman wins on  {elman_wins}/3 final-epoch validation metrics.")
    print(f"  Jordan wins on {3 - elman_wins}/3 final-epoch validation metrics.")
    print(bar + "\n")


def save_metrics_txt(final, history, save_path):
    num_epochs = len(history["elman_train_loss"])
    output = io.StringIO()

    output.write("Per-Epoch Training History\n")
    output.write("=" * 100 + "\n")
    output.write(
        f"{'Epoch':>6}  "
        f"{'E_Tr_Loss':>10}  {'E_Val_Loss':>10}  "
        f"{'E_Tr_Acc%':>10}  {'E_Val_Acc%':>10}  "
        f"{'E_Tr_PPL':>10}  {'E_Val_PPL':>10}  "
        f"{'J_Tr_Loss':>10}  {'J_Val_Loss':>10}  "
        f"{'J_Tr_Acc%':>10}  {'J_Val_Acc%':>10}  "
        f"{'J_Tr_PPL':>10}  {'J_Val_PPL':>10}\n"
    )
    output.write("-" * 100 + "\n")

    for ep in range(num_epochs):
        output.write(
            f"{ep+1:>6}  "
            f"{history['elman_train_loss'][ep]:>10.4f}  "
            f"{history['elman_val_loss'][ep]:>10.4f}  "
            f"{history['elman_train_acc'][ep]*100:>10.2f}  "
            f"{history['elman_val_acc'][ep]*100:>10.2f}  "
            f"{history['elman_train_ppl'][ep]:>10.2f}  "
            f"{history['elman_val_ppl'][ep]:>10.2f}  "
            f"{history['jordan_train_loss'][ep]:>10.4f}  "
            f"{history['jordan_val_loss'][ep]:>10.4f}  "
            f"{history['jordan_train_acc'][ep]*100:>10.2f}  "
            f"{history['jordan_val_acc'][ep]*100:>10.2f}  "
            f"{history['jordan_train_ppl'][ep]:>10.2f}  "
            f"{history['jordan_val_ppl'][ep]:>10.2f}\n"
        )

    output.write("\nFinal Metrics Summary\n")
    output.write("=" * 60 + "\n")
    for model_name, model_metrics in final.items():
        output.write(f"\n{model_name.upper()} Network\n")
        for key, val in model_metrics.items():
            formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
            output.write(f"  {key:<30} : {formatted}\n")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output.getvalue())

    print(f"  [Done] Full metrics table saved -> {save_path}")
