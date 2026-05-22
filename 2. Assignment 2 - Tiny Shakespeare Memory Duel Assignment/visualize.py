import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ELMAN_COLOR  = "#2E86AB"
JORDAN_COLOR = "#E84855"
TRAIN_ALPHA  = 1.00
VAL_ALPHA    = 0.60

FONT_TITLE = {"fontsize": 13, "fontweight": "bold"}
FONT_LABEL = {"fontsize": 11}
FONT_TICK  = {"labelsize": 10}
GRID_KW    = dict(linestyle="--", linewidth=0.5, alpha=0.5)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Done] Plot saved -> {path}")


def _moving_avg(values, window = 3):
    result = list(values)
    for i in range(window - 1, len(values)):
        result[i] = float(np.mean(values[max(0, i - window + 1): i + 1]))
    return result


def plot_loss_curves(history, save_path):
    epochs = list(range(1, len(history["elman_train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Cross-Entropy Loss - Elman vs Jordan Networks", **FONT_TITLE, y=1.02)

    for ax, model_name, color, title in [
        (axes[0], "elman",  ELMAN_COLOR,  "Elman Network"),
        (axes[1], "jordan", JORDAN_COLOR, "Jordan Network"),
    ]:
        train_loss = history[f"{model_name}_train_loss"]
        val_loss   = history[f"{model_name}_val_loss"]

        ax.plot(epochs, train_loss, color=color, lw=2.0, alpha=TRAIN_ALPHA, label="Train loss")
        ax.plot(epochs, val_loss,   color=color, lw=2.0, alpha=VAL_ALPHA, linestyle="--", label="Val loss")
        ax.plot(epochs, _moving_avg(train_loss), color=color, lw=0.8, alpha=0.4)
        ax.plot(epochs, _moving_avg(val_loss),   color=color, lw=0.8, alpha=0.3, linestyle="--")

        best_ep = int(np.argmin(val_loss))
        ax.scatter([epochs[best_ep]], [val_loss[best_ep]], color=color, s=80,
                   zorder=5, label=f"Best val (ep {epochs[best_ep]})")

        ax.set_title(title, **FONT_TITLE)
        ax.set_xlabel("Epoch", **FONT_LABEL)
        ax.set_ylabel("CE Loss (nats)", **FONT_LABEL)
        ax.tick_params(**FONT_TICK)
        ax.legend(fontsize=9)
        ax.grid(**GRID_KW)

    fig.tight_layout()
    _save(fig, save_path)


def plot_accuracy_curves(history, save_path):
    epochs = list(range(1, len(history["elman_train_acc"]) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Character-Level Accuracy - Elman vs Jordan", **FONT_TITLE)

    for model_name, color in [("elman", ELMAN_COLOR), ("jordan", JORDAN_COLOR)]:
        train_acc = [v * 100 for v in history[f"{model_name}_train_acc"]]
        val_acc   = [v * 100 for v in history[f"{model_name}_val_acc"]]
        label = model_name.capitalize()
        ax.plot(epochs, train_acc, color=color, lw=2.0, label=f"{label} Train")
        ax.plot(epochs, val_acc,   color=color, lw=2.0, linestyle="--", alpha=VAL_ALPHA, label=f"{label} Val")

    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel("Accuracy (%)", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(**GRID_KW)
    fig.tight_layout()
    _save(fig, save_path)


def plot_perplexity_curves(history, save_path):
    epochs = list(range(1, len(history["elman_val_ppl"]) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Validation Perplexity - Elman vs Jordan", **FONT_TITLE)

    for model_name, color in [("elman", ELMAN_COLOR), ("jordan", JORDAN_COLOR)]:
        ax.plot(epochs, history[f"{model_name}_val_ppl"], color=color, lw=2.2,
                label=model_name.capitalize())

    ax.axhline(y=65, color="gray", linestyle=":", lw=1.5, label="Random baseline (PPL=65)")
    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel("Perplexity (PPL)", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    ax.legend(fontsize=10)
    ax.grid(**GRID_KW)
    fig.tight_layout()
    _save(fig, save_path)


def plot_convergence_compare(history, save_path):
    epochs = list(range(1, len(history["elman_val_loss"]) + 1))
    e_val  = np.array(history["elman_val_loss"])
    j_val  = np.array(history["jordan_val_loss"])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Validation Loss Convergence Comparison", **FONT_TITLE)

    ax.plot(epochs, e_val, color=ELMAN_COLOR,  lw=2.2, label="Elman Val Loss")
    ax.plot(epochs, j_val, color=JORDAN_COLOR, lw=2.2, label="Jordan Val Loss")
    ax.fill_between(epochs, e_val, j_val, where=(e_val < j_val),
                    alpha=0.15, color=ELMAN_COLOR,  label="Elman advantage")
    ax.fill_between(epochs, e_val, j_val, where=(j_val < e_val),
                    alpha=0.15, color=JORDAN_COLOR, label="Jordan advantage")

    ax.set_xlabel("Epoch", **FONT_LABEL)
    ax.set_ylabel("Validation CE Loss (nats)", **FONT_LABEL)
    ax.tick_params(**FONT_TICK)
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)
    fig.tight_layout()
    _save(fig, save_path)


def plot_temperature_bar(results, temperatures, save_path):
    def unigram_entropy(text):
        from collections import Counter
        if not text:
            return 0.0
        counts = Counter(text)
        total  = sum(counts.values())
        probs  = [c / total for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    sorted_temps  = sorted(temperatures)
    elman_entropy  = [unigram_entropy(results[t]["elman"])  for t in sorted_temps]
    jordan_entropy = [unigram_entropy(results[t]["jordan"]) for t in sorted_temps]

    x     = np.arange(len(sorted_temps))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Effect of Temperature on Generation Diversity (Unigram Entropy)",
                 **FONT_TITLE, y=1.02)

    ax = axes[0]
    elman_bars  = ax.bar(x - width / 2, elman_entropy,  width, color=ELMAN_COLOR,  label="Elman",  alpha=0.85)
    jordan_bars = ax.bar(x + width / 2, jordan_entropy, width, color=JORDAN_COLOR, label="Jordan", alpha=0.85)

    for bar in list(elman_bars) + list(jordan_bars):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"τ={t}" for t in sorted_temps], fontsize=10)
    ax.set_ylabel("Shannon Entropy (bits)", **FONT_LABEL)
    ax.set_title("Unigram Entropy by Temperature", **FONT_TITLE)
    ax.legend(fontsize=10)
    ax.grid(axis="y", **GRID_KW)
    ax.tick_params(**FONT_TICK)

    ax2 = axes[1]
    ax2.plot(sorted_temps, elman_entropy,  "o-", color=ELMAN_COLOR,  lw=2, label="Elman")
    ax2.plot(sorted_temps, jordan_entropy, "s-", color=JORDAN_COLOR, lw=2, label="Jordan")
    ax2.set_xlabel("Temperature τ", **FONT_LABEL)
    ax2.set_ylabel("Shannon Entropy (bits)", **FONT_LABEL)
    ax2.set_title("Entropy vs Temperature (trend)", **FONT_TITLE)
    ax2.legend(fontsize=10)
    ax2.grid(**GRID_KW)
    ax2.tick_params(**FONT_TICK)

    fig.tight_layout()
    _save(fig, save_path)


def plot_dataset_overview(dataset, save_path):
    total_chars = dataset["total_chars"]
    used_chars  = dataset["used_chars"]
    discarded   = total_chars - used_chars
    train_n     = dataset["num_train"]
    val_n       = dataset["num_val"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Dataset Overview - Tiny Shakespeare", **FONT_TITLE)

    ax = axes[0]
    ax.pie(
        [used_chars, discarded],
        labels=[
            f"Used ({used_chars/total_chars*100:.0f}%)\n{used_chars:,} chars",
            f"Discarded ({discarded/total_chars*100:.0f}%)",
        ],
        colors=[ELMAN_COLOR, "#CCCCCC"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Corpus Fraction Used", **FONT_TITLE)

    ax2 = axes[1]
    bars = ax2.bar(["Train", "Validation"], [train_n, val_n],
                   color=[ELMAN_COLOR, JORDAN_COLOR], width=0.4, alpha=0.85)
    for bar, count in zip(bars, [train_n, val_n]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{count:,}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Number of Characters", **FONT_LABEL)
    ax2.set_title("Train / Validation Split", **FONT_TITLE)
    ax2.tick_params(**FONT_TICK)
    ax2.grid(axis="y", **GRID_KW)

    fig.tight_layout()
    _save(fig, save_path)


def plot_training_dashboard(history, save_path):
    epochs = list(range(1, len(history["elman_train_loss"]) + 1))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Training Dashboard - Elman vs Jordan",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    ax0 = fig.add_subplot(gs[0, 0])
    for model_name, color in [("elman", ELMAN_COLOR), ("jordan", JORDAN_COLOR)]:
        label = model_name.capitalize()
        ax0.plot(epochs, history[f"{model_name}_train_loss"], color=color, lw=1.8, label=f"{label} Train")
        ax0.plot(epochs, history[f"{model_name}_val_loss"],   color=color, lw=1.8, ls="--", alpha=0.7, label=f"{label} Val")
    ax0.set_title("Cross-Entropy Loss", **FONT_TITLE)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("CE Loss (nats)")
    ax0.legend(fontsize=8, ncol=2)
    ax0.grid(**GRID_KW)

    ax1 = fig.add_subplot(gs[0, 1])
    for model_name, color in [("elman", ELMAN_COLOR), ("jordan", JORDAN_COLOR)]:
        label = model_name.capitalize()
        ax1.plot(epochs, [v * 100 for v in history[f"{model_name}_train_acc"]], color=color, lw=1.8, label=f"{label} Train")
        ax1.plot(epochs, [v * 100 for v in history[f"{model_name}_val_acc"]],   color=color, lw=1.8, ls="--", alpha=0.7, label=f"{label} Val")
    ax1.set_title("Character Accuracy", **FONT_TITLE)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(**GRID_KW)

    ax2 = fig.add_subplot(gs[1, 0])
    for model_name, color in [("elman", ELMAN_COLOR), ("jordan", JORDAN_COLOR)]:
        ax2.plot(epochs, history[f"{model_name}_val_ppl"], color=color, lw=1.8,
                 label=f"{model_name.capitalize()} Val PPL")
    ax2.axhline(65, color="gray", ls=":", lw=1.4, label="Random (PPL=65)")
    ax2.set_title("Validation Perplexity", **FONT_TITLE)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PPL")
    ax2.legend(fontsize=8)
    ax2.grid(**GRID_KW)

    ax3 = fig.add_subplot(gs[1, 1])
    epoch_times = history.get("epoch_times", [])
    if epoch_times:
        ax3.plot(epochs[:len(epoch_times)], epoch_times, color="#7B68EE", lw=1.8,
                 marker="o", ms=4, label="Seconds / epoch")
        ax3.axhline(np.mean(epoch_times), color="#7B68EE", ls="--", alpha=0.6,
                    label=f"Mean {np.mean(epoch_times):.1f}s")
    ax3.set_title("Wall-Clock Time per Epoch", **FONT_TITLE)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Seconds")
    ax3.legend(fontsize=8)
    ax3.grid(**GRID_KW)

    _save(fig, save_path)
