

import os
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.stats import multivariate_normal
from scipy.stats import norm, laplace
import pandas as pd
import matplotlib.pyplot as plt
import random
from maf import RealNVP
from RNADE import RNADE, train_rnade, train_rnade_finale, grid_search
import matplotlib
matplotlib.use('Agg')  # backend non interattivo



#!/usr/bin/env python3


# Generazione dati senza x3, x9 e x10
# Le altre variabili rinominate in successione: x1..x7



def plot_scatter_matrix(data: np.ndarray, 
                        save_path: str = 'results/training_scatter_matrix2.png',
                        alpha: float = 0.3,
                        bins: int = 30,
                        figsize: tuple = (14, 14)) -> None:
    """
    Crea e salva una scatter matrix dei dati di training usando pandas.
    
    Parametri:
    - data: array numpy di forma (N, D)
    - save_path: percorso per salvare l'immagine
    - alpha: trasparenza dei punti
    - bins: numero di bins per gli istogrammi diagonali
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Crea DataFrame con nomi colonne
    df = pd.DataFrame(data, columns=[f"x{i+1}" for i in range(data.shape[1])])
    
    # Crea scatter matrix
    axes = pd.plotting.scatter_matrix(
        df,
        alpha=alpha,
        figsize=figsize,
        diagonal='hist',
        density_kwds={'bins': bins},
        grid=True
    )
    
    # Nascondi etichette duplicate
    for ax in axes.flatten():
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    
    plt.suptitle('Training Data Scatter Matrix', y=0.92, fontsize=16)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Scatter matrix salvata in: {save_path}")


import copy
import torch
from torch import optim

def train_realnvp(model,
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader = None,
                  lr: float = 0.0125,
                  weight_decay: float = 0,
                  epochs: int = 20,
                  device: torch.device = torch.device('cpu'),
                  patience: int = 5,
                  min_delta: float = 1e-4):
    """
    Allena un modello RealNVP con early stopping basato su val_loader.

    Parametri:
    - model: istanza di RealNVP già costruita
    - train_loader: DataLoader per il training
    - val_loader: DataLoader per la validazione (se None, nessuna early stopping)
    - lr: learning rate
    - epochs: massimo numero di epoche
    - device: torch.device su cui fare il training
    - patience: epoche consecutive senza miglioramento per stop
    - min_delta: soglia minima di miglioramento per resettare patience

    Ritorna:
    - model addestrato (ripristinando i pesi migliori su val_loader se fornito)
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_nll = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ----- training -----
        model.train()
        total_nll = 0.0
        for batch in train_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()
            nll = -model.log_prob(x).mean()
            nll.backward()
            optimizer.step()
            total_nll += nll.item() * x.size(0)

        train_nll = total_nll / len(train_loader.dataset)

        # ----- validation -----
        val_nll = None
        if val_loader is not None:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    nll = -model.log_prob(x).mean()
                    total_val += nll.item() * x.size(0)
            val_nll = total_val / len(val_loader.dataset)

            # check improvement
            if val_nll + min_delta < best_val_nll:
                best_val_nll = val_nll
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        # ----- logging -----
        if val_nll is None:
            print(f"[RealNVP] Epoch {epoch:2d}/{epochs} — Train NLL: {train_nll:.4f}")
        else:
            print(f"[RealNVP] Epoch {epoch:2d}/{epochs} — Train NLL: {train_nll:.4f} — Val NLL: {val_nll:.4f}")

        # ----- early stopping -----
        if val_loader is not None and epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs. Best Val NLL: {best_val_nll:.4f}")
            break

    # restore best weights
    if val_loader is not None:
        model.load_state_dict(best_model_wts)

    return model

def fit_multivariate_normal(train_data: np.ndarray, reg_epsilon: float = 1e-5) -> tuple:
    """
    Calcola i parametri della normale multivariata con regolarizzazione.
    
    Parametri:
    - train_data: array (N, D) con dati di training
    - reg_epsilon: valore per regolarizzazione della covarianza
    
    Ritorna:
    - mean: vettore medio (D,)
    - cov: matrice di covarianza regolarizzata (D, D)
    """
    mean = np.mean(train_data, axis=0)
    cov = np.cov(train_data, rowvar=False)
    
    # Regolarizzazione per evitare matrici singolari
    cov += reg_epsilon * np.eye(cov.shape[0])
    
    return mean, cov



    
import numpy as np
import math
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.model_selection import KFold


import numpy as np
import math
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture

def fit_gmm(data: np.ndarray,
            n_components: int = 1,
            covariance_type: str = 'full',
            random_state: int = 0):
    """
    Fitta un Gaussian Mixture Model al training set.

    Parameters
    ----------
    data : np.ndarray, shape (n,) or (n, d)
    n_components : int
        Numero di gaussiane nella mistura.
    covariance_type : str
        Tipo di covarianza, uno tra:
          - 'full': ogni componente ha la propria matrice completa di covarianza
          - 'tied': tutte le componenti condividono la stessa matrice di covarianza
          - 'diag': covarianza diagonale per ogni componente (varianze indipendenti)
          - 'spherical': covarianza sferica (un solo valore di varianza per componente)
    random_state : int

    Returns
    -------
    gmm : GaussianMixture
        Modello fittato al dataset.
    """
    X = data.reshape(-1, 1) if data.ndim == 1 else data
    gmm = GaussianMixture(n_components=n_components,
                         covariance_type=covariance_type,
                         random_state=random_state)
    gmm.fit(X)
    print(f"[fit_gmm] Fitted GMM with {n_components} components, cov_type={covariance_type}")
    return gmm


import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

def fit_kde_bandwidth_train_val(train_data: np.ndarray,
                                 val_data: np.ndarray,
                                 multipliers: np.ndarray = None,
                                 kernel: str = 'gaussian',
                                 random_state: int = 0):
    """
    Seleziona il miglior bandwidth per una stima di densità KDE usando una singola suddivisione train/validation
    e sfrutta scipy.stats.gaussian_kde come implementazione di KDE, utilizzando la regola di Scott incorporata.

    Parameters
    ----------
    train_data   : np.ndarray, shape (n_train,) or (n_train, d)
        Dati di training per il fitting del KDE.
    val_data     : np.ndarray, shape (n_val,) or (n_val, d)
        Dati di validation per la selezione del bandwidth.
    multipliers  : np.ndarray of float, optional
        Fattori moltiplicativi applicati al bandwidth pilota; se None, usa np.logspace(-1, 1, 50).
    kernel       : str, default 'gaussian'
        Tipo di kernel (supportato solo 'gaussian').
    random_state : int, default 0
        Seed per la riproducibilità (non usato direttamente).

    Returns
    -------
    best_kde : gaussian_kde
        Oggetto gaussian_kde fitted sui dati di training con il miglior bandwidth.
    best_bw  : float
        Valore di bw_method ottimale (scalar factor).
    """
    # Assicura array 2D
    X_tr = train_data.reshape(-1, 1) if train_data.ndim == 1 else train_data
    X_val = val_data.reshape(-1, 1)   if val_data.ndim == 1   else val_data

    # Stima il bandwidth pilota usando la regola di Scott incorporata in scipy
    kde_pilot = gaussian_kde(X_tr.T, bw_method='scott')
    pilot_factor = kde_pilot.covariance_factor()

    # Definisci la griglia di moltiplicatori
    if multipliers is None:
        multipliers = np.logspace(-1, 1, 100)
    factors = multipliers * pilot_factor

    # Valutazione su validation set: massimizza log-likelihood medio
    val_scores = []
    for factor in factors:
        kde = gaussian_kde(X_tr.T, bw_method=factor)
        logp = kde.logpdf(X_val.T)
        val_scores.append(np.mean(logp))

    # Selezione del miglior fattore
    best_idx = int(np.argmax(val_scores))
    best_factor = factors[best_idx]

    # Fitting finale
    best_kde = gaussian_kde(X_tr.T, bw_method=best_factor)

    print(f"[fit_kde_bandwidth_train_val] Pilot factor: {pilot_factor:.4f}, "
          f"Best multiplier: {multipliers[best_idx]:.3f}, Best factor: {best_factor:.4f}")
    return best_kde, best_factor



def fit_gmm_train_val(train_data: np.ndarray,
                      val_data: np.ndarray,
                      n_components_list: list = None,
                      covariance_type: str = 'full',
                      random_state: int = 0):
    """
    Seleziona il miglior numero di componenti per un GMM usando un train/validation split.

    Parameters
    ----------
    train_data        : np.ndarray, shape (n_train,) or (n_train, d)
    val_data          : np.ndarray, shape (n_val,)   or (n_val, d)
    n_components_list : list of int, optional
        Candidate numbers of gaussians; if None, uses [1,2,...,9].
    covariance_type   : str
        One of {'full','tied','diag','spherical'}.
    random_state      : int
        Random seed for reproducibility.

    Returns
    -------
    best_gmm : GaussianMixture
        GMM fitted on the full training set with best number of components.
    best_n   : int
        The selected number of components.
    """
    X_tr  = train_data.reshape(-1, 1) if train_data.ndim == 1 else train_data
    X_val = val_data.reshape(-1, 1)   if val_data.ndim == 1   else val_data

    if n_components_list is None:
        n_components_list = list(range(1, 30))

    val_scores = []
    for n in n_components_list:
        gmm = GaussianMixture(n_components=n,
                              covariance_type=covariance_type,
                              random_state=random_state).fit(X_tr)
        val_scores.append(gmm.score(X_val))  # total log-likelihood

    best_idx = int(np.argmax(val_scores))
    best_n   = n_components_list[best_idx]
    best_gmm = GaussianMixture(n_components=best_n,
                               covariance_type=covariance_type,
                               random_state=random_state).fit(X_tr)

    print(f"[fit_gmm_train_val] Best n_components: {best_n}")
    return best_gmm, best_n




    



def fit_gmm_cv(data: np.ndarray,
               n_components_list: list = None,
              covariance_type: str = 'full',
               cv_folds: int = 5,
               random_state: int = 0):
    """
    Seleziona il miglior numero di componenti per un GMM via cross-validation.

    Parameters
    ----------
    data : np.ndarray, shape (n,) or (n, d)
   n_components_list : list of int, optional
        Lista di valori di n_components da testare. Se None, usa [1, 2, 3, 4, 5].
    covariance_type : str
        Tipo di covarianza per tutti i modelli.
   cv_folds : int
        Numero di fold per la validazione incrociata.
    random_state : int

    Returns
    -------
    best_gmm : GaussianMixture
        Modello GMM fittato sull'intero dataset con n_components ottimale.
    best_n : int
       Numero di componenti ottimale.
    """
    X = data.reshape(-1, 1) if data.ndim == 1 else data
    if n_components_list is None:
        n_components_list = list(range(1, 10))
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    mean_scores = []
    for n in n_components_list:
        scores = []
        for tr, te in kf.split(X):
            gmm = GaussianMixture(n_components=n,
                                  covariance_type=covariance_type,
                                  random_state=random_state)
            gmm.fit(X[tr])
            scores.append(gmm.score(X[te]))
        mean_scores.append(np.mean(scores))
    best_idx = int(np.argmax(mean_scores))
    best_n = n_components_list[best_idx]
    best_gmm = GaussianMixture(n_components=best_n,
                               covariance_type=covariance_type,
                               random_state=random_state).fit(X)
    print(f"[fit_gmm_cv] Best n_components: {best_n}")
    return best_gmm, best_n

    


def plot_density_comparisons(estimates: list, true_density: np.ndarray, dim: int, out_dir: str = 'results', dataset: str = None):
    """
    Generates a separate plot for each density estimate.

    Parameters
    ----------
    estimates : list of tuples
        List of (name: str, est_density: np.ndarray) for the estimates.
    true_density : np.ndarray, shape (n,)
        Theoretical density on the test data.
    dim : int
        Dimensionality of the data.
    out_dir : str
        Directory to save the plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Indices for sorting according to the true density
    order = np.argsort(true_density)

    for name, est in estimates:

        plt.figure(figsize=(8, 4))
        # Plot estimated density first
        plt.plot(est[order], label=f'{name}', zorder=1)
        # Plot true density on top
        plt.plot(true_density[order], label='True density', linewidth=2, zorder=2)
        plt.xlabel('Samples sorted by true density')
        plt.ylabel('Density')
        plt.title(f"Density Comparison (dim={dim})")
        plt.legend()
        plt.tight_layout()

        # Save the plot with dimensionality in the filename
        fig_path = os.path.join(out_dir, f'density_comparison_{name}_dim{dim}_dataset{dataset}.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved density comparison plot for {name} to {fig_path}")

def plot_density_comparisons_filter(estimates: list, true_density: np.ndarray, dim: int, out_dir: str = 'results', dataset: str = None):
    """
    Generates a separate plot for each density estimate, considering only points where true_density < 10.

    Parameters
    ----------
    estimates : list of tuples
        List of (name: str, est_density: np.ndarray) for the estimates.
    true_density : np.ndarray, shape (n,)
        Theoretical density on the test data.
    dim : int
        Dimensionality of the data.
    out_dir : str
        Directory to save the plots.
    dataset : str or None
        Optional dataset name for filenames.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Filter indices where true_density < 0.1
    valid_idx = np.where(true_density < 0.1)[0]

    # Sort these filtered indices by true_density
    order = valid_idx[np.argsort(true_density[valid_idx])]

    for name, est in estimates:
        plt.figure(figsize=(8, 4))
        # Plot only filtered and sorted densities
        plt.plot(est[order], label=f'{name}', zorder=1)
        plt.plot(true_density[order], label='True density', linewidth=2, zorder=2)
        plt.xlabel('Samples sorted by true density (true_density < .1)')
        plt.ylabel('Density')
        plt.title(f"Density Comparison (dim={dim})")
        plt.legend()
        plt.tight_layout()

        # Construct filename safely even if dataset is None
        dataset_str = dataset if dataset is not None else 'unknown'
        fig_path = os.path.join(out_dir, f'density_comparison_{name}_dim{dim}_dataset{dataset_str}.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved density comparison plot for {name} to {fig_path}")       

def rnade_est(test_data: np.ndarray,
              model: torch.nn.Module) :
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32)
        logp = model(xt).cpu().numpy()
    est_density = np.exp(logp)

    return est_density

def realnvp_est(test_data: np.ndarray,
                model: torch.nn.Module) :
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32, device=device)
        logp = model.log_prob(xt).cpu().numpy()
    est_density = np.exp(logp)

    return est_density

def gmm_est(test_data: np.ndarray,
            gmm_model: GaussianMixture) :
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = gmm_model.score_samples(X_test)
    est = np.exp(logp)
    return est

def kde_est(test_data: np.ndarray,
            kde_model: KernelDensity) :
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = kde_model.logpdf(X_test.T)
    est = np.exp(logp)
    return est


def kde_est(test_data: np.ndarray, kde_model) -> np.ndarray:
    """
    Restituisce densità stimate su test_data usando il modello KDEProduct.
    """
    # Riusiamo score_samples
    return kde_model.score_samples(test_data)

def evaluate_rmse(est_density: np.ndarray, true_density: np.ndarray) :
    rmse = np.sqrt(np.mean((est_density - true_density)**2))
    return rmse
def evaluate_rmse_median(est_density: np.ndarray, true_density: np.ndarray) :
    rmse = np.sqrt(np.median((est_density - true_density)**2))
    return rmse

def evaluate_ise(est_density: np.ndarray, true_density: np.ndarray) :
    ise = np.mean(((est_density - true_density)**2)/true_density)
    return ise
def evaluate_ise_median(est_density: np.ndarray, true_density: np.ndarray) :
    ise = np.median(((est_density - true_density)**2)/true_density)
    return ise

def evaluate_rel_L1(est_density: np.ndarray, true_density: np.ndarray) :
    rel_L1 = np.mean(np.abs(est_density - true_density)/true_density)
    return rel_L1

def evaluate_rel_L1_median(est_density: np.ndarray, true_density: np.ndarray) -> float:
    rel_errors = np.abs(est_density - true_density) / true_density
    rel_L1_median = np.median(rel_errors)
    return rel_L1_median

def evaluate_rnade_rmse(test_data: np.ndarray,
                   model: torch.nn.Module,
                   true_density: np.ndarray) :
    """
    Calcola l'RMSE tra densità teorica e densità stimata da RNADE sul test set,
    ritorna RMSE e densità stimata.
    """
    # 2) densità stimata da RNADE
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32)
        logp = model(xt).cpu().numpy()
    est_density = np.exp(logp)

    # 3) calcolo RMSE
    rmse = np.sqrt(np.mean((est_density - true_density)**2))
    print(f"RMSE RNADE: {rmse:.6f}")
    return rmse, est_density
    '''
    ise = np.mean(((est_density - true_density)**2)/true_density)
    return ise, est_density
    '''

def evaluate_realnvp_rmse(test_data: np.ndarray,
                    model: torch.nn.Module,
                    true_density: np.ndarray) :
    """
    Calcola l'RMSE tra densità teorica e RealNVP, ritorna RMSE e densità stimata.
    """
    # 2) densità stimata da RealNVP
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32, device=device)
        logp = model.log_prob(xt).cpu().numpy()
    est_density = np.exp(logp)

    # 3) calcolo RMSE
    rmse = np.sqrt(np.mean((est_density - true_density) ** 2))
    print(f"RMSE RealNVP: {rmse:.6e}")
    return rmse, est_density

def evaluate_gmm_rmse(test_data: np.ndarray,
                      gmm_model: GaussianMixture,
                      true_density: np.ndarray):
    """
    Calcola RMSE tra densità vera e densità stimata da GMM.

    Parameters
    ----------
    test_data : np.ndarray, shape (n,) or (n, d)
    gmm_model : GaussianMixture (già fittato)
   true_density : np.ndarray, shape (n,)
    Returns
    -------
    rmse : float
    est_density : np.ndarray
   """
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = gmm_model.score_samples(X_test)
    est = np.exp(logp)
    rmse = np.sqrt(np.mean((est - true_density)**2))
    print(f"[evaluate_gmm_rmse] RMSE: {rmse:.6e}")
    return rmse, est

def evaluate_kde_rmse(test_data: np.ndarray,
                      kde_model,
                      true_density: np.ndarray):
    """
    Calcola RMSE densità da KDE.
    """
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = kde_model.logpdf(X_test.T)
    est = np.exp(logp)
    rmse = np.sqrt(np.mean((est - true_density)**2))
    print(f"[evaluate_kde_rmse] RMSE: {rmse:.6e}")
    return rmse, est

def evaluate_gmm_ise(test_data: np.ndarray,
                      gmm_model: GaussianMixture,
                      true_density: np.ndarray):
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = gmm_model.score_samples(X_test)
    est = np.exp(logp)
    ise = np.mean(((est - true_density)**2)/true_density)
    return ise, est


def evaluate_kde_ise(test_data: np.ndarray,
                      kde_model,
                      true_density: np.ndarray):
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = kde_model.logpdf(X_test.T)
    est = np.exp(logp)
    ise = np.mean(((est - true_density)**2)/true_density)
    return ise, est


def evaluate_realnvp_ise(test_data: np.ndarray,
                    model: torch.nn.Module,
                    true_density: np.ndarray) :
    # 2) densità stimata da RealNVP
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32, device=device)
        logp = model.log_prob(xt).cpu().numpy()
    est_density = np.exp(logp)

    ise = np.mean(((est_density - true_density)**2)/true_density)
    return ise, est_density

def evaluate_rnade_ise(test_data: np.ndarray,
                   model: torch.nn.Module,
                   true_density: np.ndarray) :
    # 2) densità stimata da RNADE
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32)
        logp = model(xt).cpu().numpy()
    est_density = np.exp(logp)
    errors = ((est_density - true_density) ** 2) / true_density

    # 3) salva su CSV
    df = pd.DataFrame({
        "est_density": est_density,
        "true_density": true_density,
        "weighted_error": errors
    })
    df.to_csv('errors.csv', index=False)

    # 4) calcola ISE medio
    ise = np.mean(errors)
    return ise, est_density
    

def evaluate_gmm_rel_L1(test_data: np.ndarray,
                      gmm_model: GaussianMixture,
                      true_density: np.ndarray):
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = gmm_model.score_samples(X_test)
    est = np.exp(logp)
    ise = np.mean(((est - true_density)**2)/true_density)
    return ise, est


def evaluate_kde_rel_L1(test_data: np.ndarray,
                      kde_model,
                      true_density: np.ndarray):
    X_test = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
    logp = kde_model.logpdf(X_test.T)
    est = np.exp(logp)
    ise = np.mean(((est - true_density)**2)/true_density)
    return ise, est


def evaluate_realnvp_rel_L1(test_data: np.ndarray,
                    model: torch.nn.Module,
                    true_density: np.ndarray) :
    # 2) densità stimata da RealNVP
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32, device=device)
        logp = model.log_prob(xt).cpu().numpy()
    est_density = np.exp(logp)

    ise = np.mean(((est_density - true_density)**2)/true_density)
    return ise, est_density

def evaluate_rnade_rel_L1(test_data: np.ndarray,
                   model: torch.nn.Module,
                   true_density: np.ndarray) :
    # 2) densità stimata da RNADE
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(test_data, dtype=torch.float32)
        logp = model(xt).cpu().numpy()
    est_density = np.exp(logp)
    errors = ((est_density - true_density) ** 2) / true_density

    # 3) salva su CSV
    df = pd.DataFrame({
        "est_density": est_density,
        "true_density": true_density,
        "weighted_error": errors
    })
    df.to_csv('errors.csv', index=False)

    # 4) calcola ISE medio
    ise = np.mean(errors)
    return ise, est_density

