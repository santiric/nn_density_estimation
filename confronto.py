import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Import dei modelli definiti nei file forniti
from RNADE import RNADE, train_rnade
from realnvp import RealNVP, create_masks, train_flow


def generate_complex_data(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x1 = np.random.randn(N)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    # Eliminazione della variabile x3
    mu2, sd2 = np.sin(2 * x1), 0.5 + 0.1 * np.abs(np.sin(x1)); x2 = np.random.normal(mu2, sd2)
    mu4, sd4 = np.cos(x1), 0.3 + 0.05 * np.abs(np.cos(x1));   x4 = np.random.normal(mu4, sd4)
    mu5, sd5 = x1 * np.sin(x1), 0.5 + 0.2 * np.abs(x1);       x5 = np.random.normal(mu5, sd5)
    mu6, sd6 = np.tanh(x1), 0.4 + 0.1 * x1**2;               x6 = np.random.normal(mu6, sd6)
    mu7, sd7 = np.log(np.abs(x1) + 1), 0.3 + 0.1 * np.abs(x1); x7 = np.random.normal(mu7, sd7)
    mu8, sd8 = np.exp(-0.2 * x1), 0.6;                       x8 = np.random.normal(mu8, sd8)
    mu9, sd9 = (x1**2) * np.sin(x1), 0.5 + 0.2 * np.abs(np.sin(x1)); x9 = np.random.normal(mu9, sd9)

    w10 = sigmoid(3 * np.sin(x1))
    comp1 = np.random.normal(x1, 1.0)
    comp2 = np.random.normal(-x1, 1.5)
    u = np.random.rand(N)
    x10 = np.where(u < w10, comp1, comp2)

    cols = [f"x{i}" for i in range(1, 11) if i != 3]  # Rimuove la colonna x3
    return pd.DataFrame({c: arr for c, arr in zip(cols, [x1, x2, x4, x5, x6, x7, x8, x9, x10])})

def theoretical_log_prob(X):
    data = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    x1, *rest = data.T
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    lp = norm(0, 1).logpdf(x1)

    for i, (func, base_sd) in enumerate([
        (lambda x: np.sin(2 * x), lambda x: 0.5 + 0.1 * np.abs(np.sin(x))),
        (lambda x: np.cos(x), lambda x: 0.3 + 0.05 * np.abs(np.cos(x))),
        (lambda x: x * np.sin(x), lambda x: 0.5 + 0.2 * np.abs(x)),
        (lambda x: np.tanh(x), lambda x: 0.4 + 0.1 * x**2),
        (lambda x: np.log(np.abs(x) + 1), lambda x: 0.3 + 0.1 * np.abs(x)),
        (lambda x: np.exp(-0.2 * x), lambda x: 0.6 * np.ones_like(x)),
        (lambda x: (x**2) * np.sin(x), lambda x: 0.5 + 0.2 * np.abs(np.sin(x)))
    ], start=2):
        xi = rest[i - 2]
        mu = func(x1)
        sd = base_sd(x1)
        lp += norm(mu, sd).logpdf(xi)

    # Componente 10
    x10 = rest[-1]
    w10 = sigmoid(3 * np.sin(x1))
    p1 = norm(x1, 1.0).pdf(x10)
    p2 = norm(-x1, 1.5).pdf(x10)
    lp += np.log(w10 * p1 + (1 - w10) * p2)

    return lp




def main():
    # Generazione dati
    N = 5000
    df = generate_complex_data(N, seed=123)
    print(f"Shape del dataset: {df.shape}")
    print(f"Prime righe:\n{df.head()}")

    # Controllo su ogni colonna del dataset
    for col in df.columns:
        col_data = df[col]
        print(f"\nColonna: {col}")
        print(f"  Range: {col_data.min()} to {col_data.max()}")
        print(f"  Mean: {col_data.mean():.4f}, Std: {col_data.std():.4f}")
        if col_data.isna().any():
            print(f"  Contiene NaN!")
        if (col_data.abs() > 1e6).any():
            print(f"  Valori estremi (>1e6) trovati!")


    # Split raw
    train_data, temp = train_test_split(df.values, test_size=0.3, random_state=42)
    valid_data, test_data = train_test_split(temp, test_size=0.5, random_state=42)

    # ===== RNADE su tutte e 10 le dimensioni =====
    
    rnade = RNADE(input_dim=9, hidden_units=100, num_components=8)
    rnade = train_rnade(
        rnade, train_data, valid_data,
        num_epochs=400, batch_size=128,
        init_lr=0.05, weight_decay=0.001, patience=30
    )
    

    # ===== RealNVP su tutte e 10 le dimensioni =====
    D = train_data.shape[1]
    hidden_dim = 500
    n_layers = 20
    mask = create_masks(D, n_layers)
    # disabilitiamo il controllo di supporto per evitare errori
    prior = torch.distributions.MultivariateNormal(
        loc=torch.zeros(D),
        covariance_matrix=torch.eye(D),
        validate_args=True
    )
    realnvp = RealNVP(D, hidden_dim, n_layers, mask)

    realnvp = train_flow(
        realnvp,
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(valid_data, dtype=torch.float32),
        batch_size=256, lr=1e-3,
        epochs=200, patience=20,
        device='cpu'
    )

    # ===== Valutazione su Test Set =====
    rnade.eval(); realnvp.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_data, dtype=torch.float32)
        logp_rnade = rnade(test_tensor).cpu().numpy()
        logp_flow  = realnvp.log_prob(test_tensor).cpu().numpy()

    density_rnade = np.exp(logp_rnade)
    density_flow  = np.exp(logp_flow)
    density_theory = np.exp(theoretical_log_prob(test_data))

    # ===== Plot confronto densità =====
    idx = np.argsort(density_theory)
    plt.figure(figsize=(12, 6))
    plt.plot(density_theory[idx], label="Teorica", linewidth=2)
    plt.plot(density_rnade[idx], label="RNADE (10 dim)", linestyle="--")
    plt.plot(density_flow[idx],  label="RealNVP (10 dim)", linestyle="-.")
    plt.xlabel("Esempi ordinati per densità teorica crescente")
    plt.ylabel("Densità")
    plt.title("Confronto: Teorica vs RNADE vs RealNVP su dati complessi")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


