
import pickle, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print('Loading predictions...')
with open('/root/mtl4rec/results/predictions/SharedBottom_base_predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)
print(f'Loaded {predictions["n_samples"]} samples')

TASK_NAMES = ['is_click', 'is_like', 'is_follow', 'is_comment',
              'is_forward', 'long_view', 'is_hate', 'is_profile_enter']

# Prepare data
X = np.column_stack([predictions['y_pred'][t] for t in TASK_NAMES])
print(f'X shape: {X.shape}')

# Compute relevance
def compute_relevance_hierarchical(y_true_dict):
    n = len(y_true_dict[TASK_NAMES[0]])
    pos_count = np.zeros(n)
    for task in TASK_NAMES:
        if task == 'is_hate':
            continue
        pos_count += y_true_dict[task]
    has_follow_or_comment = (y_true_dict['is_follow'] + y_true_dict['is_comment']) > 0
    has_like_or_forward_or_long = (y_true_dict['is_like'] + y_true_dict['is_forward'] + y_true_dict['long_view']) > 0
    has_click_or_profile = (y_true_dict['is_click'] + y_true_dict['is_profile_enter']) > 0
    has_hate = y_true_dict['is_hate'] > 0
    relevance = np.where(has_follow_or_comment, 300,
                np.where(has_like_or_forward_or_long, 200,
                np.where(has_click_or_profile, 100, 0)))
    relevance = relevance.astype(np.float64) + pos_count
    relevance = np.where(has_hate, 0, relevance)
    return relevance

y = compute_relevance_hierarchical(predictions['y_true'])
print(f'y shape: {y.shape}, range: [{y.min():.1f}, {y.max():.1f}]')

# Subset for training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {X_train.shape}, Val: {X_val.shape}')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val)

print('Creating model...')
class FusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = FusionMLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print('Training...')
for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: loss={loss.item():.4f}')

print('Done!')
