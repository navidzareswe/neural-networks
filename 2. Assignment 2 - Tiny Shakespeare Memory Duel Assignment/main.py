import os
import time
import numpy as np

import visualize
from metrics import compute_final_metrics, print_final_metrics_report, save_metrics_txt
from generate import temperature_experiment, print_generation_report
from train import train
from models import ElmanRNN, JordanRNN
from data_loader import load_and_prepare

DATA_FRACTION = 0.05
TRAIN_RATIO   = 0.90
SEQ_LENGTH    = 50
HIDDEN_SIZE   = 128
LEARNING_RATE = 1e-2
CLIP_VALUE    = 5.0
NUM_EPOCHS    = 12

SEED_TEXT    = "ROMEO: "
GEN_LENGTH   = 300
TEMPERATURES = [0.4, 0.8, 1.2]

DATA_ZIP   = os.path.join(os.path.dirname(__file__), "data", "tiny-shakespeare.zip")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
GEN_DIR    = os.path.join(OUTPUT_DIR, "generated")

np.random.seed(42)


def main():
    pipeline_start = time.time()

    _print_banner()
    _ensure_dirs()

    print("\n[STEP 1/7]  Loading and preprocessing dataset...")
    dataset = load_and_prepare(
        zip_path=DATA_ZIP,
        data_fraction=DATA_FRACTION,
        train_ratio=TRAIN_RATIO,
        seed=42,
    )
    _print_dataset_stats(dataset)

    print("\n[STEP 2/7]  Instantiating models...")
    elman = ElmanRNN(
        vocab_size=dataset["vocab_size"],
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        clip_val=CLIP_VALUE,
    )
    jordan = JordanRNN(
        vocab_size=dataset["vocab_size"],
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        clip_val=CLIP_VALUE,
    )
    print(f"  {elman}")
    print(f"  {jordan}")

    print("\n[STEP 3/7]  Generating dataset overview...")
    visualize.plot_dataset_overview(
        dataset=dataset,
        save_path=os.path.join(PLOT_DIR, "01_dataset_overview.png"),
    )

    print("\n[STEP 4/7]  Training both models simultaneously...")
    history = train(
        elman=elman,
        jordan=jordan,
        dataset=dataset,
        seq_length=SEQ_LENGTH,
        num_epochs=NUM_EPOCHS,
    )

    print("\n[STEP 5/7]  Computing final evaluation metrics...")
    final_metrics = compute_final_metrics(history)
    print_final_metrics_report(final_metrics, total_epochs=NUM_EPOCHS)
    save_metrics_txt(final_metrics, history, os.path.join(OUTPUT_DIR, "metrics.txt"))

    print("\n[STEP 6/7]  Generating text at multiple temperatures...")
    char2idx = dataset["char2idx"]
    idx2char = dataset["idx2char"]

    gen_results = temperature_experiment(
        elman=elman,
        jordan=jordan,
        seed_text=SEED_TEXT,
        char2idx=char2idx,
        idx2char=idx2char,
        temperatures=TEMPERATURES,
        gen_length=GEN_LENGTH,
    )
    print_generation_report(SEED_TEXT, gen_results, save_path=os.path.join(GEN_DIR, "generation_report.txt"))

    for temp, texts in gen_results.items():
        for model_name, text in texts.items():
            out_path = os.path.join(GEN_DIR, f"{model_name}_tau{temp:.1f}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Model    : {model_name.capitalize()}\n")
                f.write(f"Seed     : {repr(SEED_TEXT)}\n")
                f.write(f"Temp (τ) : {temp}\n")
                f.write(f"Length   : {GEN_LENGTH} chars\n")
                f.write("-" * 60 + "\n\n")
                f.write(text)
            print(f" Saved -> {out_path}")

    print("\n[STEP 7/7]  Generating all plots...")
    visualize.plot_loss_curves(history,         save_path=os.path.join(PLOT_DIR, "02_loss_curves.png"))
    visualize.plot_accuracy_curves(history,     save_path=os.path.join(PLOT_DIR, "03_accuracy_curves.png"))
    visualize.plot_perplexity_curves(history,   save_path=os.path.join(PLOT_DIR, "04_perplexity_curves.png"))
    visualize.plot_convergence_compare(history, save_path=os.path.join(PLOT_DIR, "05_convergence_comparison.png"))
    visualize.plot_temperature_bar(
        results=gen_results,
        temperatures=TEMPERATURES,
        save_path=os.path.join(PLOT_DIR, "06_temperature_diversity.png"),
    )
    visualize.plot_training_dashboard(history, save_path=os.path.join(PLOT_DIR, "08_training_dashboard.png"))

    _print_completion_summary(time.time() - pipeline_start)


def _print_banner():
    print("=" * 72)
    print("  NEURAL NETWORKS AND DEEP LEARNING (2026) - ASSIGNMENT 2")
    print("  Tiny Shakespeare Memory Duel")
    print("  Elman RNN  vs  Jordan RNN  -  Character-Level Language Modelling")
    print("=" * 72)
    print(f"\n  Hyperparameters")
    print(f"    Data fraction  : {DATA_FRACTION*100:.0f}% of corpus")
    print(f"    Seq length     : {SEQ_LENGTH}")
    print(f"    Hidden size    : {HIDDEN_SIZE}")
    print(f"    Learning rate  : {LEARNING_RATE}")
    print(f"    Clip value     : {CLIP_VALUE}")
    print(f"    Epochs         : {NUM_EPOCHS}")
    print(f"    Seed text      : {repr(SEED_TEXT)}")
    print(f"    Gen length     : {GEN_LENGTH}")
    print(f"    Temperatures   : {TEMPERATURES}")


def _print_dataset_stats(dataset):
    print(f"\n  Dataset Statistics")
    print(f"    Total corpus chars   : {dataset['total_chars']:>10,}")
    print(f"    Fraction used        : {dataset['data_fraction']*100:.0f}%")
    print(f"    Characters used      : {dataset['used_chars']:>10,}")
    print(f"    Training chars       : {dataset['num_train']:>10,}")
    print(f"    Validation chars     : {dataset['num_val']:>10,}")
    print(f"    Vocabulary size      : {dataset['vocab_size']:>10}")
    print(f"\n  Dataset sample (first 200 chars):")
    print("  " + repr(dataset["text_sample"][:200]))


def _ensure_dirs():
    for directory in [OUTPUT_DIR, PLOT_DIR, GEN_DIR]:
        os.makedirs(directory, exist_ok=True)


def _print_completion_summary(total_time):
    m, s = divmod(int(total_time), 60)
    print("\n" + "=" * 72)
    print("  EXPERIMENT COMPLETE")
    print(f"  Total wall-clock time : {m}m {s}s")
    print(f"\n  Output files:")
    print(f"    Plots     -> output/plots/   (7 figures)")
    print(f"    Generated -> output/generated/")
    print(f"    Metrics   -> output/metrics.txt")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
