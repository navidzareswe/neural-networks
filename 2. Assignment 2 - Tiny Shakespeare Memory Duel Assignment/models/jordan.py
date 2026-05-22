import numpy as np


class JordanRNN:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        learning_rate = 1e-2,
        clip_val = 5.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.clip_val = clip_val

        rng = np.random.default_rng(1)
        self.W_xh = rng.uniform(-0.1, 0.1, (self.hidden_size, self.vocab_size))
        self.W_yh = rng.uniform(-0.1, 0.1, (self.hidden_size, self.vocab_size))
        self.W_hy = rng.uniform(-0.1, 0.1, (self.vocab_size, self.hidden_size))
        self.b_h  = np.zeros((self.hidden_size, 1))
        self.b_y  = np.zeros((self.vocab_size, 1))

        self.param_names = ["W_xh", "W_yh", "W_hy", "b_h", "b_y"]

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        return exp_vals / (exp_vals.sum() + 1e-9)

    def forward(self, inputs, h_prev, p_prev):
        one_hot, hiddens, probs = {}, {}, {}
        hiddens[-1] = h_prev.copy()
        probs[-1]   = p_prev.copy()

        for t, char_idx in enumerate(inputs):
            one_hot[t] = np.zeros((self.vocab_size, 1))
            one_hot[t][char_idx] = 1.0

            hiddens[t] = np.tanh(
                self.W_xh @ one_hot[t]
                + self.W_yh @ probs[t - 1]
                + self.b_h
            )
            logits   = self.W_hy @ hiddens[t] + self.b_y
            probs[t] = self._softmax(logits)

        return one_hot, hiddens, probs

    def backward(self, one_hot, hiddens, probs, targets):
        seq_len = len(targets)

        grad_W_xh = np.zeros_like(self.W_xh)
        grad_W_yh = np.zeros_like(self.W_yh)
        grad_W_hy = np.zeros_like(self.W_hy)
        grad_b_h  = np.zeros_like(self.b_h)
        grad_b_y  = np.zeros_like(self.b_y)

        total_loss = 0.0

        for t in reversed(range(seq_len)):
            total_loss += -np.log(probs[t][targets[t], 0] + 1e-9)

            out_delta = probs[t].copy()
            out_delta[targets[t]] -= 1.0

            grad_W_hy += out_delta @ hiddens[t].T
            grad_b_y  += out_delta

            delta_h = self.W_hy.T @ out_delta
            grad_h  = (1.0 - hiddens[t] ** 2) * delta_h

            grad_b_h  += grad_h
            grad_W_xh += grad_h @ one_hot[t].T
            grad_W_yh += grad_h @ probs[t - 1].T

        grads = dict(W_xh=grad_W_xh, W_yh=grad_W_yh, W_hy=grad_W_hy, b_h=grad_b_h, b_y=grad_b_y)
        for g in grads.values():
            np.clip(g, -self.clip_val, self.clip_val, out=g)

        return total_loss / seq_len, grads

    def sgd_update(self, grads):
        for name in self.param_names:
            getattr(self, name).__isub__(self.lr * grads[name])

    def train_step(
        self,
        inputs,
        targets,
        h_prev,
        p_prev,
    ):
        one_hot, hiddens, probs = self.forward(inputs, h_prev, p_prev)
        loss, grads = self.backward(one_hot, hiddens, probs, targets)
        self.sgd_update(grads)
        seq_len = len(inputs)
        preds = [int(np.argmax(probs[t])) for t in range(seq_len)]
        return loss, hiddens[seq_len - 1], probs[seq_len - 1], preds

    def evaluate_step(
        self,
        inputs,
        targets,
        h_prev,
        p_prev,
    ):
        one_hot, hiddens, probs = self.forward(inputs, h_prev, p_prev)
        seq_len = len(inputs)
        loss = sum(-np.log(probs[t][targets[t], 0] + 1e-9) for t in range(seq_len))
        preds = [int(np.argmax(probs[t])) for t in range(seq_len)]
        return loss / seq_len, hiddens[seq_len - 1], probs[seq_len - 1], preds

    def generate(
        self,
        seed_indices,
        h_init,
        p_init,
        length,
        temperature = 1.0,
    ):
        output_indices = list(seed_indices)
        hidden = h_init.copy()
        out_probs = p_init.copy()
        temp = max(temperature, 1e-8)

        for char_idx in seed_indices:
            x = np.zeros((self.vocab_size, 1))
            x[char_idx] = 1.0
            hidden    = np.tanh(self.W_xh @ x + self.W_yh @ out_probs + self.b_h)
            logits    = self.W_hy @ hidden + self.b_y
            out_probs = self._softmax(logits)

        curr_idx = seed_indices[-1]
        for _ in range(length):
            x = np.zeros((self.vocab_size, 1))
            x[curr_idx] = 1.0
            hidden        = np.tanh(self.W_xh @ x + self.W_yh @ out_probs + self.b_h)
            logits        = self.W_hy @ hidden + self.b_y
            scaled_logits = logits / temp - np.max(logits / temp)
            out_probs     = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
            curr_idx      = np.random.choice(self.vocab_size, p=out_probs.ravel())
            output_indices.append(curr_idx)

        return output_indices

    def count_parameters(self):
        return sum(getattr(self, name).size for name in self.param_names)

    def __repr__(self):
        return (
            f"JordanRNN(vocab={self.vocab_size}, hidden={self.hidden_size}, "
            f"params={self.count_parameters():,}, lr={self.lr})"
        )
