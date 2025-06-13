import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Config
DATA_PATH = "data/dataset.csv"
BATCH_SIZE = 64
TOTAL_STEPS = 15000
EVAL_EVERY = 100  # pasos entre evaluaciones
LEARNING_RATE = 0.0009
TOLERANCE = 0.05

# Leer datos
df = pd.read_csv(DATA_PATH)

# One-hot del operador
op_encoder = OneHotEncoder(sparse_output=False)
op_encoded = op_encoder.fit_transform(df[['operator']])

# Inputs: numerador, operador, denominador
X = np.concatenate([
    df[['numerator']].values,
    op_encoded,
    df[['denominator']].values
], axis=1).astype(np.float32)

y = df['result'].values.astype(np.float32).reshape(-1, 1)

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo
class CalculatorNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = CalculatorNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Métricas para graficar
train_losses = []
val_losses = []
val_maes = []
val_accuracies = []
eval_steps = []

# Entrenamiento por STEP
step = 0
model.train()
train_iterator = iter(train_loader)

while step < TOTAL_STEPS:
    try:
        inputs, targets = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        inputs, targets = next(train_iterator)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if step % EVAL_EVERY == 0 or step == TOTAL_STEPS - 1:
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_tensor).numpy()
            val_targets = y_test_tensor.numpy()

            val_loss = criterion(torch.tensor(val_preds), y_test_tensor).item()
            mae = mean_absolute_error(val_targets, val_preds)
            relative_errors = np.abs(val_preds - val_targets) / (np.abs(val_targets) + 1e-8)
            acc = np.mean(relative_errors < TOLERANCE)

            val_losses.append(val_loss)
            val_maes.append(mae)
            val_accuracies.append(acc)
            eval_steps.append(step)

        print(f"Step {step}/{TOTAL_STEPS} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} | MAE: {mae:.4f} | Accuracy (@{TOLERANCE*100:.0f}% tol): {acc*100:.2f}%")
        model.train()

    step += 1

# Guardar modelo
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/calculator_model.pth")
print("\nModelo guardado en models/calculator_model.pth")

# Post-entrenamiento
os.makedirs("output", exist_ok=True)

# 1. Loss Curve + MAE
# 1. Loss Curve + MAE (con twin axes)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Eje izquierdo: Loss
ax1.plot(train_losses, label="Train Loss (per step)", alpha=0.5)
ax1.plot(eval_steps, val_losses, label="Validation Loss (on eval)", color='red')
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.tick_params(axis='y')
ax1.grid(True)

# Eje derecho: MAE
ax2 = ax1.twinx()
ax2.plot(eval_steps, val_maes, label="Validation MAE", color='green')
ax2.set_ylabel("MAE")
ax2.tick_params(axis='y')

# Unir leyendas de ambos ejes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.title("Training Progress: Loss & MAE")
plt.tight_layout()
plt.savefig("output/loss_curve.png")
plt.close()


# 2. Accuracy per Operator
op_labels = op_encoder.get_feature_names_out(['operator'])
df_op = pd.DataFrame(op_encoded, columns=op_labels)
df_input = pd.DataFrame(df[['numerator', 'denominator']].values, columns=['numerator', 'denominator'])
df_full = pd.concat([df_input, df_op], axis=1)

df_full["operator"] = df_op.idxmax(axis=1).str.extract(r"operator_(.*)")[0]
df_full["true"] = y.flatten()
df_full["predicted"] = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()

acc_by_op = df_full.groupby("operator").apply(
    lambda g: np.mean(np.abs(g["true"] - g["predicted"]) / (np.abs(g["true"]) + 1e-8) < TOLERANCE)
)

ax = acc_by_op.plot(kind="bar", color="skyblue", edgecolor="black")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy per Operator")
ax.set_ylim(0, 1)
ax.grid(axis="y")

# Solución para el label "-"
ax.set_xticklabels(acc_by_op.index, rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig("output/accuracy_per_operator.png")
plt.close()

# 3. Error Distribution
errors = (df_full["true"] - df_full["predicted"]).values
sns.histplot(errors, bins=10, kde=True, color="lightcoral")
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("output/error_distribution.png")
plt.close()


# 4. Scatter plot: valores reales vs predichos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_full["predicted"], y=errors, alpha=0.6, color="purple", edgecolors="black", s=20)
plt.axhline(y=0, color='red', linestyle='--', label='Zero Error Line')
plt.xlabel("Predicted Value")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Plot: Predicted Values vs. Residuals")
plt.legend()
plt.grid(True)
plt.savefig("output/residual_plot.png")
plt.close()