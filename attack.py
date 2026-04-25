import numpy as np
import pandas as pd

# Adversarial attack (perturbation)
def adversarial_attack(X, epsilon=0.2):
    noise = np.random.normal(0, epsilon, X.shape)
    return X + noise

# Data poisoning
def poison_data(df, poison_rate=0.3):
    df_poisoned = df.copy()
    n = int(len(df) * poison_rate)

    indices = np.random.choice(df.index, n)

    # flip labels
    df_poisoned.loc[indices, 'label'] = 1 - df_poisoned.loc[indices, 'label']

    return df_poisoned