# autopep8: off
import numpy as np
from visulazie import print_weights_and_biases_22, print_summary_22, plot_energy_landscape_22
from hopfield_sr import HopfieldSR

SR_CASES = [
    (1, -1, [1., -1.], "Set    <-- from Set   state"),
    (1, -1, [-1.,  1.], "Set    <-- from Reset state"),
    (1, -1, [-1., -1.], "Set    <-- from invalid-"),
    (1, -1, [1.,  1.], "Set    <-- from invalid+"),

    (-1,  1, [-1.,  1.], "Reset  <-- from Reset state"),
    (-1,  1, [1., -1.], "Reset  <-- from Set   state"),
    (-1,  1, [-1., -1.], "Reset  <-- from invalid-"),
    (-1,  1, [1.,  1.], "Reset  <-- from invalid+"),# [oscillation; expected]
    (-1, -1, [1., -1.], "Hold/Memory   <-- retain Set  state"),# [oscillation; expected]
    (-1, -1, [-1.,  1.], "Hold/Memory   <-- retain Reset state"),
    (-1, -1, [1.,  1.], "Hold/Memory   <-- from invalid+"),
    (-1, -1, [-1., -1.], "Hold/Memory   <-- from invalid-"),

    (1,  1, [1., -1.], "Invalid   <-- from Set   [indeterminate]"),
    (1,  1, [-1.,  1.], "Invalid   <-- from Reset [indeterminate]"),# [oscillation; expected]
    (1,  1, [1.,  1.], "Invalid   <-- from invalid+"),# [oscillation; expected]
    (1,  1, [-1., -1.], "Invalid   <-- from invalid-"),
]


def run_functional_tests(hopfield_net):
    print("\n\n  <--- Functional SR Latch Tests <---\n")
    print((f"  {'S':>3} {'R':>3} | {'Init State':>13} | {'Final State':>13} | "
           f"{'Iters':>5} | {'Conv?':>5} | {'Valid?':>5} | Condition"))
    print("  " + "-" * 90)

    func_valid = []
    for S, R, init_state_list, description in SR_CASES:
        initial_state = np.array(init_state_list)
        final_state, iterations, is_converged, _ = hopfield_net.run(
            S, R, initial_state)

        is_valid = hopfield_net.is_valid_state(final_state)
        func_valid.append(is_valid)

        conv_status = "yes" if is_converged else " NO"
        valid_status = "yes" if is_valid else " NO"
        init_str = str([int(v) for v in initial_state])
        final_str = str([int(v) for v in final_state])
        print(f"  {S:+3d} {R:+3d} | {init_str:>13} | "
              f"{final_str:>13} | {iterations:>5} | "
              f"{conv_status:>5} | {valid_status:>5} | {description}")

    return func_valid


def run_noise_tests(hopfield_net):
    print("\n\n  <--- Noise Robustness Tests (single-bit flips) <---\n")
    print(f"  {'SR Cond':>9} | {'Base Pattern':>14} | {'Flip':>6} | "
          f"{'Noisy Input':>13} | {'Final State':>13} | {'Conv':>5} | {'Valid':>5}")
    print("  " + "-" * 86)

    stored_patterns = [np.array([1., -1.]),  np.array([-1.,  1.])]
    pattern_labels = ["Set  [+1,-1]",        "Reset[-1,+1]"]
    test_conditions = [(1, -1, "Set"), (-1, 1, "Reset"), (-1, -1, "Hold")]

    noise_test_results = []
    for S, R, condition_name in test_conditions:
        for base_pattern, pattern_label in zip(stored_patterns, pattern_labels):
            for flip_idx in range(hopfield_net.N):
                noisy_input = base_pattern.copy()
                noisy_input[flip_idx] *= -1.0

                final_state, _, is_converged, _ = hopfield_net.run(
                    S, R, noisy_input)

                is_valid = hopfield_net.is_valid_state(final_state)
                noise_test_results.append(is_valid)

                conv_status = "yes" if is_converged else " NO"
                valid_status = "yes" if is_valid else " NO"
                noisy_str = str([int(v) for v in noisy_input])
                final_str = str([int(v) for v in final_state])

                print(f"  {condition_name:>9} | {pattern_label:>14} | bit[{flip_idx}] | "
                      f"{noisy_str:>13} | "
                      f"{final_str:>13} | {conv_status:>5} | {valid_status:>5}")
        print()

    return noise_test_results


def run_section_22():
    print("=" * 70)
    print("  PART 2.2 - Hopfield Network: SR Latch")
    print("=" * 70)

    hopfield_net = HopfieldSR()

    print_weights_and_biases_22(hopfield_net)

    func_valid = run_functional_tests(hopfield_net)
    noise_valid = run_noise_tests(hopfield_net)

    print_summary_22(func_valid, noise_valid)
    plot_energy_landscape_22(hopfield_net)


if __name__ == '__main__':
    run_section_22()
