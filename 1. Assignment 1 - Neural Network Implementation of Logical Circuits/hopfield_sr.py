import numpy as np
from activation_functions import sign_activation

SET = np.array([1.0, -1.0])
RESET = np.array([-1.0, 1.0])


class HopfieldSR:
    def __init__(self):
        self.N = 2
        self.patterns = np.array([SET, RESET])
        S = self.patterns.T
        self.W = np.zeros((self.N, self.N))

        self.W = S @ S.T
        # Preventing Huge Numbers - sgn logic doesn't change
        self.W = self.W / self.N
        np.fill_diagonal(self.W, 0.0)

    def get_external_signal(self, S, R):
        """
        Calculates the external signal vector x = [S-R, R-S].
        This maps binary S/R inputs to bipolar forces for the Hopfield neurons:

        1. S=0, R=0 (Memory/Hold):  x = [ 0,  0]
        2. S=1, R=0 (SET):          x = [ 1, -1] 
        3. S=0, R=1 (RESET):        x = [-1,  1]
        4. S=1, R=1 (INVALID):      x = [ 0,  0] -> Inputs cancel out; safely defaults to HOLD.
        """
        return np.array([float(S) - float(R), float(R) - float(S)])

    def calc_energy(self, y, S, R):
        x = self.get_external_signal(S, R)

        internode_energy = -0.5 * (y.T @ self.W @ y)
        external_energy = -(x @ y)

        return float(internode_energy + external_energy)

    def check_convergence(self, y_new, y):
        return np.array_equal(y_new, y)

    def run(self, S, R, y0, max_iter=50):
        y = y0.copy().astype(float)
        x = self.get_external_signal(S, R)
        history = [y.copy()]

        for i in range(max_iter):
            y_in = (self.W @ y) + x
            y_new = np.array([sign_activation(y_in_i, y_i)
                              for y_in_i, y_i in zip(y_in, y)])

            if self.check_convergence(y_new, y):
                return y_new, i + 1, True, history

            # Update current state for the next iteration
            y = y_new.copy()
            history.append(y.copy())

        return y, max_iter, False, history

    def is_valid_state(self, s):
        set_pattern = self.patterns[0]
        reset_pattern = self.patterns[1]

        is_set = np.array_equal(s, set_pattern)
        is_reset = np.array_equal(s, reset_pattern)

        return is_set or is_reset

    def state_name(self, s):
        if np.array_equal(s, SET):
            return "Set   [+1, -1]"
        elif np.array_equal(s, RESET):
            return "Reset [-1, +1]"
        else:
            return f"Indeterminate [{int(s[0])}, {int(s[1])}]"
