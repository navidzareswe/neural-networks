from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from activation_functions import bi_sigmoid, bi_sigmoid_derivative, bip_sigmoid, bip_sigmoid_derivative
from visulazie import plot_mse_21, print_truth_table_21, print_comparison_table_21

np.random.seed(42)


def f1(A, B, C):
    # out = (A & ~B & ~C) | (~A & B & ~C) | (~A & ~B & C) | (A & B & C)
    return int(
        (A and not B and not C) or
        (not A and B and not C) or
        (not A and not B and C) or
        (A and B and C)
    )


def f2(A, B, C, D):
    # out = (A & B & C) | (A & B & D) | (A & C &D) | (B & C & D)

    return int(
        (A and B and C) or
        (A and B and D) or
        (A and C and D) or
        (B and C and D)
    )


def build_dataset(n_inputs, func, bipolar=False):
    combos = list(product([0, 1], repeat=n_inputs))

    X = np.array(combos, dtype=float)
    y = np.array([func(*row) for row in combos], dtype=float)

    if bipolar:
        X = 2.0 * X - 1.0
        y = 2.0 * y - 1.0
    return X, y


def train_perceptron(X, t, bipolar=False, alpha=0.1, tolerance=1e-5, n_epochs=2000):
    if bipolar:
        activation_fn, activation_derivative, threshold = bip_sigmoid, bip_sigmoid_derivative, 0.0
    else:
        activation_fn, activation_derivative, threshold = bi_sigmoid, bi_sigmoid_derivative, 0.5

    samples, feat = X.shape
    weights = np.random.uniform(-0.1, 0.1, feat)
    bias = np.random.uniform(-0.1, 0.1)

    max_weight_change = float('inf')
    ep = 0
    mse_history = []
    first_perfect_epoch = None

    while max_weight_change > tolerance and ep < n_epochs:
        sse = 0.0
        max_weight_change = 0.0
        for i in range(samples):
            y_in = float(X[i] @ weights) + bias
            y = activation_fn(y_in)

            err = t[i] - y
            delta = err * activation_derivative(y)
            delta_w = alpha * delta * X[i]
            delta_b = alpha * delta

            weights += delta_w
            bias += delta_b

            sse += err ** 2

            current_max_change = max(np.max(np.abs(delta_w)), abs(delta_b))
            if current_max_change > max_weight_change:
                max_weight_change = current_max_change

        mse = sse / samples
        mse_history.append(mse)

        # Check accuracy this epoch -> vectorised for efficiency
        out_all = activation_fn(X @ weights + bias)
        if bipolar:
            preds_ep = np.where(out_all >= 0.0, 1.0, -1.0)
        else:
            preds_ep = (out_all >= threshold).astype(float)

        if np.all(preds_ep == t) and first_perfect_epoch is None:
            first_perfect_epoch = ep + 1
        ep += 1

    # Final predictions
    out_f = activation_fn(X @ weights + bias)
    if bipolar:
        predictions = np.where(out_f >= 0.0, 1.0, -1.0)
    else:
        predictions = (out_f >= threshold).astype(float)
    accuracy = float(np.mean(predictions == t))

    return weights, bias, accuracy, np.array(mse_history), first_perfect_epoch, predictions


def run_task_representation(fname, func, nin, rep_name, is_bip, ax):
    print(f"\n{'-' * 64}")
    print(f"  {fname}   [{rep_name.upper()} representation]")
    print(f"{'-' * 64}")

    X, y = build_dataset(nin, func, bipolar=is_bip)
    W, b, acc, mse_log, perf_ep, preds = train_perceptron(
        X, y, bipolar=is_bip, alpha=0.1, n_epochs=2000
    )

    print_truth_table_21(X, y, preds)

    print(f"\n  Weights  : {np.round(W, 4).tolist()}")
    print(f"  Bias     : {b:.4f}")
    print(f"  Accuracy : {acc * 100:.1f}%")
    if perf_ep:
        print(f"  First epoch with 100% accuracy: {perf_ep}")
    else:
        print(f"  Did NOT reach 100% accuracy  "
              f"(final MSE = {mse_log[-1]:.4e})")

    # Plot
    plot_mse_21(ax, mse_log, perf_ep,
                f"{fname}\n({rep_name} representation)",
                is_bip)

    return acc, mse_log, perf_ep


def run_section_21():
    print("=" * 70)
    print("  PART 2.1 - Multi-Input Logic Gate Classification")
    print("=" * 70)

    TASKS = [
        ("F1: 3-input Odd Parity (XOR)", f1, 3),
        ("F2: 4-input Majority-3-of-4",  f2, 4),
    ]
    REPRESENTATIONS = [("binary", False), ("bipolar", True)]

    fig_mse, axes_mse = plt.subplots(2, 2, figsize=(13, 9))
    fig_mse.suptitle("Delta Learning Rule - MSE per Epoch",
                     fontsize=13, fontweight="bold")

    results = {}

    for fn_idx, (fname, func, nin) in enumerate(TASKS):
        for rep_idx, (rep_name, is_bip) in enumerate(REPRESENTATIONS):

            acc, mse_log, perf_ep = run_task_representation(
                fname, func, nin,
                rep_name, is_bip,
                axes_mse[fn_idx, rep_idx]
            )

            results[(fn_idx, rep_idx)] = (acc, perf_ep)

    plt.tight_layout()
    plt.savefig("part21_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  [Figure saved: part21_convergence.png]")

    print_comparison_table_21(TASKS, REPRESENTATIONS, results)


if __name__ == '__main__':
    run_section_21()
