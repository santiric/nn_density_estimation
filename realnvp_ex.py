#.\.venv\Scripts\Activate.ps1
"""
Train RealNVP on a 10D correlated Gaussian (analytic density available), then test on 1,000 fresh samples.
Save as train_realnvp_10d.py alongside maf.py and run via:
    python train_realnvp_10d.py
"""
#https://github.com/kamenbliznashki/normalizing_flows
'''
paper di riferimento: @article{bnaf19,
  title={Block Neural Autoregressive Flow},
  author={De Cao, Nicola and
          Titov, Ivan and
          Aziz, Wilker},
  journal={35th Conference on Uncertainty in Artificial Intelligence (UAI19)},
  year={2019}
}'''
import os
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# import your RealNVP implementation
from maf import RealNVP


class CorrelatedGaussianDataset(Dataset):
    """
    10D Gaussian with covariance cov[i,j] = rho**|i-j|.
    Pre-samples n_samples points and provides analytic log-density.
    """
    def __init__(self, n_samples=100_000, dim=10, rho=0.8, device=None):
        self.dim = dim
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)

        # build mean & covariance
        idx = torch.arange(dim)
        cov = rho ** (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mean = torch.zeros(dim)

        # PyTorch dist object
        self.dist = torch.distributions.MultivariateNormal(mean, cov)
        # pre-sample data
        self.data = self.dist.sample((n_samples,)).to(self.device)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        return self.data[i]

    def log_density(self, x):
        """Returns log p(x) under the true Gaussian (x: [...,dim])."""
        return self.dist.log_prob(x)


def train_and_test_realnvp(
    n_samples=100_000,
    dim=10,
    rho=0.8,
    batch_size=256,
    n_blocks=6,
    hidden_size=128,
    n_hidden=2,
    lr=1e-3,
    epochs=20,
    test_samples=1000
):
    # 1) prepare dataset & loader
    dataset = CorrelatedGaussianDataset(
        n_samples=n_samples, dim=dim, rho=rho
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2) build RealNVP
    model = RealNVP(
        n_blocks=n_blocks,
        input_size=dim,
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        batch_norm=True
    ).to(dataset.device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3) training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_nll = 0.0
        for x in loader:
            x = x.to(dataset.device)
            optimizer.zero_grad()
            # model.log_prob returns log p_model(x)
            nll = - model.log_prob(x).mean()
            nll.backward()
            optimizer.step()
            total_nll += nll.item() * x.size(0)

        avg_nll = total_nll / len(dataset)
        # true NLL under target dist (constant)
        with torch.no_grad():
            true_nll = - dataset.log_density(dataset.data).mean().item()

        print(f"Epoch {epoch:2d}/{epochs} — NLL(model): {avg_nll:.4f} | NLL(true): {true_nll:.4f}")

    # 4) testing on fresh samples
    print("\nTesting on 1,000 new samples...")
    x_test = dataset.dist.sample((test_samples,)).to(dataset.device)
    with torch.no_grad():
        logp_est = model.log_prob(x_test)
        logp_th = dataset.log_density(x_test)
    p_est = torch.exp(logp_est).cpu().numpy()
    p_th = torch.exp(logp_th).cpu().numpy()

    # print table of estimated vs theoretical
    df = pd.DataFrame({
        'estimated': p_est,
        'theoretical': p_th
    })
    print(df.to_string(index=False))

    # plot sorted densities
    order = p_th.argsort()
    plt.figure(figsize=(8,4))
    plt.plot(p_th[order], label='Theoretical density')
    plt.plot(p_est[order], label='Estimated density')
    plt.xlabel('Samples sorted by true density')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    # Save the comparison plot since Agg backend is non-interactive
    out_dir = 'results'
    if not os.path.isdir(out_dir): os.makedirs(out_dir)
    fig_path = os.path.join(out_dir, 'density_comparison.png')
    plt.savefig(fig_path)
    print(f"Saved density comparison plot to {fig_path}")
    plt.close()


if __name__ == "__main__":
    train_and_test_realnvp(
        n_samples=100_000,
        dim=10,
        rho=0.8,
        batch_size=256,
        n_blocks=6,
        hidden_size=128,
        n_hidden=2,
        lr=1e-3,
        epochs=10,
        test_samples=1000
    )