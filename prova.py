import numpy as np
import torch
from torch import nn, distributions
from torch.utils.data import DataLoader, TensorDataset
import copy

# 1) Funzione per generare maschere a blocchi alternati
def create_masks(D, n_layers):
    d1 = D // 2
    d2 = D - d1
    masks = [
        np.array([1]*d1 + [0]*d2, dtype=np.float32) if i % 2 == 0
        else np.array([0]*d1 + [1]*d2, dtype=np.float32)
        for i in range(n_layers)
    ]
    return torch.from_numpy(np.stack(masks))

# 2) Definizione del modello RealNVP generalizzato
class RealNVP(nn.Module):
    def __init__(self, D, hidden_dim, n_layers, mask, prior):
        super().__init__()
        self.D     = D
        self.prior = prior
        self.mask  = nn.Parameter(mask, requires_grad=False)
        # reti di scala e traslazione
        self.s = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, D)
            ) for _ in range(n_layers)
        ])
        self.t = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
                nn.Linear(hidden_dim, D)
            ) for _ in range(n_layers)
        ])

    def g(self, z):
        x = z
        for i, (s_net, t_net) in enumerate(zip(self.s, self.t)):
            m  = self.mask[i]
            x_ = x * m
            s  = s_net(x_) * (1 - m)
            t  = t_net(x_) * (1 - m)
            x  = x_ + (1 - m) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J = torch.zeros(x.shape[0], device=x.device)
        z = x
        for i in reversed(range(len(self.s))):
            m  = self.mask[i]
            z_ = z * m
            s  = self.s[i](z_) * (1 - m)
            t  = self.t[i](z_) * (1 - m)
            z  = (1 - m) * ((z - t) * torch.exp(-s)) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z) + log_det_J

    def sample(self, batch_size):
        z = self.prior.sample((batch_size,))
        return self.g(z)

# 3) Funzione di training con early stopping


def train_flow(
    flow: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int = 300,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    device: str = 'cpu'
):
    """
    Allena un RealNVP con early stopping e validation set esterno.

    Parametri:
    - flow: istanza di RealNVP
    - train_data: tensor [N_train, D] per il training
    - val_data: tensor [N_val, D] per la validazione
    - batch_size, lr, epochs, patience: hyperparametri
    - device: 'cpu' o 'cuda'
    """
    flow.to(device)
    
    # Sposta i dati sul dispositivo corretto
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # Crea il DataLoader per il training
    train_loader = DataLoader(
        TensorDataset(train_data), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    for ep in range(epochs):
        # Training
        flow.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            loss = -flow.log_prob(batch_x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        flow.eval()
        with torch.no_grad():
            val_loss = -flow.log_prob(val_data).mean().item()

        print(f'Epoca {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(flow.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoca {ep} (no improve in {patience})')
                flow.load_state_dict(best_state)
                break

    # Ripristina il miglior modello trovato
    if best_state is not None:
        flow.load_state_dict(best_state)
        
    return flow

if __name__ == '__main__':
    # Configurazione
    D = 5
    n_layers = 20
    hidden_dim = 128
    SEED = 42
    torch.manual_seed(SEED)

    # 1. Generazione dati
    comp_means = torch.tensor([[2.0]*D, [-2.0]*D])
    components = distributions.MultivariateNormal(
        comp_means,
        covariance_matrix=torch.eye(D).unsqueeze(0).expand(2, D, D)
    )
    mixing = distributions.Categorical(torch.tensor([0.5, 0.5]))
    mixture = distributions.MixtureSameFamily(mixing, components)

    # Genera dataset completo
    full_data = mixture.sample((3000,))
    
    # 2. Split in train/val
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        full_data,
        test_size=0.1,
        random_state=SEED
    )

    # 3. Inizializzazione modello
    masks = create_masks(D, n_layers)
    prior = distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
    flow = RealNVP(D, hidden_dim, n_layers, masks, prior)

    # 4. Addestramento con validation set esplicito
    trained_flow = train_flow(
        flow,
        train_data=train_data,
        val_data=val_data,  # Nuovo parametro
        batch_size=300,
        lr=1e-3,
        epochs=100,
        patience=10,
        device='cpu'
    )

    # 5. Valutazione su nuovo test set
    x_test = mixture.sample((10,))
    true_lp = mixture.log_prob(x_test)
    est_lp = trained_flow.log_prob(x_test)

    print("\nConfronto log-probabilità (True vs Estimated):")
    for i in range(10):
        print(f"Osservazione {i}: True={true_lp[i]:.4f}, Flow={est_lp[i]:.4f}")