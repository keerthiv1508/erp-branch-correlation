import json
import numpy as np
with open('training_data.json') as f:
    data = json.load(f)
X = np.array([np.array(d['x']).reshape(2, 75) for d in data], dtype=np.float32)
y = np.array([d['y_label'] for d in data], dtype=np.float32)
groups = np.array([d['sample_index'] for d in data])
np.savez('lstm_data_by_program.npz', X=X, y=y, groups=groups)
print(f"Saved {len(y)} pairs, {len(set(groups))} programs")
print(f"X shape: {X.shape}, y shape: {y.shape}")
