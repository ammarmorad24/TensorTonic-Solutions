import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    if rng is None:
        r = np.random.random()
        if r < epsilon:
            return int(np.random.randint(len(q_values)))
        else:
            return int(np.argmax(q_values))
    else:
        r = rng.random()
        if r < epsilon:
            return int(rng.integers(len(q_values)))
        else:
            return int(np.argmax(q_values))