import numpy as np
from sklearn.linear_model import Ridge

class ReservoirTrainer:
    """
    Handles training of the linear readout layer using Ridge Regression.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def train(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples, n_targets)
        """
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def collect_features(reservoir, inputs, sim_steps, washout_steps):
    """
    Run reservoir and collect features (e.g., averaged voltages or spike counts).
    """
    n_neurons = reservoir.n_neurons
    features = []
    
    for t in range(sim_steps):
        v, s = reservoir.step(inputs[t])
        if t >= washout_steps:
            # Using membrane voltage as feature (standard for HH RC)
            features.append(v.copy())
            
    return np.array(features)
