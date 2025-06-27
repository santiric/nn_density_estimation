GLOBAL_SEED = 986

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
from functions import(plot_scatter_matrix,train_realnvp,fit_multivariate_normal,
    fit_kde_bandwidth_train_val,fit_gmm,fit_gmm_train_val,plot_density_comparisons,
    gmm_est,kde_est,rnade_est,realnvp_est,evaluate_ise,evaluate_rmse,evaluate_rel_L1,
    plot_density_comparisons_filter,evaluate_rel_L1_median,evaluate_ise_median,evaluate_rmse_median)

#multivariate normal; autoregressive structure

def ar1_cor(d: int, rho: float) -> np.ndarray:
    """
    Genera la matrice di correlazione AR(1) di dimensione d x d con parametro rho.

    Parametri
    ----------
    d   : int
        Dimensione della matrice (numero di variabili).
    rho : float
        Coefficiente di correlazione autoregressiva (|rho| < 1).

    Ritorna
    -------
    cor_matrix : np.ndarray, shape (d, d)
        Matrice di correlazione AR(1).
    """
    # Creiamo un array 0:(d-1)
    idx = np.arange(d)
    # Calcoliamo la distanza assoluta tra indici
    exponent = np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1))
    # Eleviamo rho alle potenze corrispondenti
    cor_matrix = rho ** exponent
    return cor_matrix

def generate_data(N: int,
                              seed: int,
                              sigma: np.ndarray,
                              mean: np.ndarray = None) -> np.ndarray:
    """
    Genera N campioni da una normale multivariata con media e matrice di covarianza sigma
    e applica la trasformazione Y = exp(0.05 * X) elemento per elemento.

    Parametri
    ----------
    N     : int
        Numero di campioni da generare.
    seed  : int
        Seed per il generatore di numeri casuali.
    sigma : np.ndarray, shape (d, d)
        Matrice di covarianza (definita positiva).
    mean  : np.ndarray, shape (d,), facoltativo
        Vettore di medie; default vettore di zeri.

    Ritorna
    -------
    Y : np.ndarray, shape (N, d)
        Matrice dei campioni trasformati.
    """
    # Seed per riproducibilità
    rng = np.random.default_rng(seed)
    # Dimensione
    d = sigma.shape[0]
    if mean is None:
        mean = np.zeros(d)
    # Genera X multivariata
    X = rng.multivariate_normal(mean, sigma, size=N)
    # Applica la trasformazione
    Y = np.exp(0.7*X)
    return Y


def density(Y: np.ndarray,
                        sigma: np.ndarray,
                        mean: np.ndarray = None) -> np.ndarray:
    """
    Calcola la densità dei punti Y dopo la trasformazione Y = exp(0.05 * X)
    partendo dalla densità di X ~ N(mean, sigma).
    Utilizza la formula del cambio di variabile:
    f_Y(y) = f_X(g^{-1}(y)) * |det(D g^{-1}(y))|

    Parametri
    ----------
    Y     : np.ndarray, shape (n_samples, d)
        Punti trasformati in cui valutare la densità.
    sigma : np.ndarray, shape (d, d)
        Matrice di covarianza di X.
    mean  : np.ndarray, shape (d,), facoltativo
        Vettore di medie di X; default vettore di zeri.

    Ritorna
    -------
    densities_Y : np.ndarray, shape (n_samples,)
        Valori di densità per ciascun punto Y.
    """
    # Imposta media di X
    d = sigma.shape[0]
    if mean is None:
        mean = np.zeros(d)
    # Inversa della trasformazione: X = (1/0.05) * ln(Y)
    X = (1 / 0.7) * np.log(Y)
    # Calcola densità di X
    mvn = multivariate_normal(mean=mean, cov=sigma)
    f_X = mvn.pdf(X)
    # Calcola Jacobiana assoluta: prod_i (1 / (0.05 * y_i))
    # Per ogni campione: det = (1 / 0.05)^d * prod_i (1 / y_i)
    jacobian_det = ((1 / 0.7) ** d) * np.prod(1 / Y, axis=1)
    # Densità di Y
    f_Y = f_X * jacobian_det
    return f_Y
def set_seed(seed: int) -> None:
    """
    Imposta i seed per random, numpy e torch per garantire la riproducibilità.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Funzione per inizializzare i worker dei DataLoader (picklable)
def worker_init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)
    random.seed(GLOBAL_SEED + worker_id)
def main():
    set_seed(GLOBAL_SEED)
    N = 25000
    D=10
    sigma = ar1_cor(D, 0.9)
    data = generate_data(N, GLOBAL_SEED, sigma)
    
    # Split train/valid/test
    train_data, temp_data = train_test_split(
        data, test_size=21/25, random_state=GLOBAL_SEED
    )
    valid_data, test_data = train_test_split(
        temp_data, test_size=20/21, random_state=GLOBAL_SEED
    )
    

    print(f"Dimensione del set di addestramento: {train_data.shape}")
    print(f"Dimensione del set di validazione: {valid_data.shape}")
    print(f"Dimensione del set di test: {test_data.shape}")

    # Calcolo densità vera sul TEST prima dello shuffle delle colonne
    true_density = density(test_data, sigma)
       
    # Genera permutazione delle colonne e shuffle dei dataset
    rng = np.random.RandomState(GLOBAL_SEED)
    n_features = train_data.shape[1]
    perm = rng.permutation(n_features)
    train_data = train_data[:, perm]
    valid_data = valid_data[:, perm]
    test_data  = test_data[:,  perm]

    # DataLoader PyTorch (needed for RealNVP)
    set_seed(GLOBAL_SEED)
    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_data, dtype=torch.float32)),
        batch_size=256, shuffle=True,
        generator=generator, worker_init_fn=worker_init_fn, num_workers=4
    )

    set_seed(GLOBAL_SEED)
    generator_val = torch.Generator()
    generator_val.manual_seed(GLOBAL_SEED)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(valid_data, dtype=torch.float32)),
        batch_size=256, shuffle=True,
        generator=generator_val, worker_init_fn=worker_init_fn, num_workers=4
    )


    # Statistiche descrittive sul TRAIN permutato
    set_seed(GLOBAL_SEED)
    mean_per_feature = np.mean(train_data, axis=0)
    variance_per_feature = np.var(train_data, axis=0)
    for i, (mean_f, var_f) in enumerate(zip(mean_per_feature, variance_per_feature)):
        print(f"Feature {i+1}: media = {mean_f:.4f}, varianza = {var_f:.4f}")


    # --- GMM ---
    set_seed(GLOBAL_SEED)
    gmm_model, best_n = fit_gmm_train_val(
        train_data, valid_data,
        n_components_list=[1,3,5,7,10,15,20,25,30,35,40],
        random_state=GLOBAL_SEED
    )
    
    # --- KDE ---
    set_seed(GLOBAL_SEED)
    kde_model, bw = fit_kde_bandwidth_train_val(
        train_data, valid_data, random_state=GLOBAL_SEED
    )

    # --- RNADE ---
    set_seed(GLOBAL_SEED)
    hidden_units = 50
    rnade = RNADE(train_data.shape[1], hidden_units, 10)
    rnade = train_rnade(
        rnade, train_data, valid_data,
        num_epochs=2000, batch_size=256,
        init_lr=0.005, weight_decay=10e-5,
        patience=50
    )

    # --- RealNVP ---
    set_seed(GLOBAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realnvp = RealNVP(
        n_blocks=10,
        input_size=train_data.shape[1],
        hidden_size=128,
        n_hidden=3,
        batch_norm=True
    )
    realnvp = train_realnvp(
        realnvp, train_loader, val_loader=val_loader,
        lr=1e-3, epochs=60,
        device=device, patience=5, min_delta=1e-4
    )

    # --- Valutazioni finali ---
    

    est_rnade = rnade_est(test_data, rnade)
    est_realnvp = realnvp_est(test_data, realnvp)
    est_kde = kde_est(test_data, kde_model)
    est_gmm = gmm_est(test_data, gmm_model)
    
    rnade_rmse = evaluate_rmse(est_rnade, true_density)
    realnvp_rmse = evaluate_rmse(est_realnvp, true_density)
    kde_rmse = evaluate_rmse(est_kde, true_density)
    gmm_rmse = evaluate_rmse(est_gmm, true_density)

    # Riepilogo RMSE
    print("\nRiepilogo RMSE:")
    print(f"- RNADE:                {rnade_rmse:.3e}")
    print(f"- RealNVP:              {realnvp_rmse:.3e}")
    print(f"- KDE:                  {kde_rmse:.3e}")
    print(f"- GMM:                  {gmm_rmse:.3e}\n")


    rnade_ise = evaluate_ise(est_rnade, true_density)
    realnvp_ise = evaluate_ise(est_realnvp, true_density)
    kde_ise = evaluate_ise(est_kde, true_density)
    gmm_ise = evaluate_ise(est_gmm, true_density)

    # Riepilogo ISE
    print("\nRiepilogo ISE:")
    print(f"- RNADE:                {rnade_ise:.3e}")
    print(f"- RealNVP:              {realnvp_ise:.3e}")
    print(f"- KDE:                  {kde_ise:.3e}")
    print(f"- GMM:                  {gmm_ise:.3e}\n")
   

    rnade_rel_L1 = evaluate_rel_L1(est_rnade, true_density)
    realnvp_rel_L1 = evaluate_rel_L1(est_realnvp, true_density)
    kde_rel_L1 = evaluate_rel_L1(est_kde, true_density)
    gmm_rel_L1 = evaluate_rel_L1(est_gmm, true_density)

    # Riepilogo rel_L1
    print("\nRiepilogo rel_L1:")
    print(f"- RNADE:                {rnade_rel_L1:.3e}")
    print(f"- RealNVP:              {realnvp_rel_L1:.3e}")
    print(f"- KDE:                  {kde_rel_L1:.3e}")
    print(f"- GMM:                  {gmm_rel_L1:.3e}\n")

    rnade_rmse_median = evaluate_rmse_median(est_rnade, true_density)
    realnvp_rmse_median = evaluate_rmse_median(est_realnvp, true_density)
    kde_rmse_median = evaluate_rmse_median(est_kde, true_density)
    gmm_rmse_median = evaluate_rmse_median(est_gmm, true_density)

    # Riepilogo RMSE_median
    print("\nRiepilogo RMSE_median:")
    print(f"- RNADE:                {rnade_rmse_median:.3e}")
    print(f"- RealNVP:              {realnvp_rmse_median:.3e}")
    print(f"- KDE:                  {kde_rmse_median:.3e}")
    print(f"- GMM:                  {gmm_rmse_median:.3e}\n")

    rnade_ise_median = evaluate_ise_median(est_rnade, true_density)
    realnvp_ise_median = evaluate_ise_median(est_realnvp, true_density)
    kde_ise_median = evaluate_ise_median(est_kde, true_density)
    gmm_ise_median = evaluate_ise_median(est_gmm, true_density)

    # Riepilogo ISE_median
    print("\nRiepilogo ISE_median:")
    print(f"- RNADE:                {rnade_ise_median:.3e}")
    print(f"- RealNVP:              {realnvp_ise_median:.3e}")
    print(f"- KDE:                  {kde_ise_median:.3e}")
    print(f"- GMM:                  {gmm_ise_median:.3e}\n")

    rnade_rel_L1_median = evaluate_rel_L1_median(est_rnade, true_density)
    realnvp_rel_L1_median = evaluate_rel_L1_median(est_realnvp, true_density)
    kde_rel_L1_median = evaluate_rel_L1_median(est_kde, true_density)
    gmm_rel_L1_median = evaluate_rel_L1_median(est_gmm, true_density)

    # Riepilogo rel_L1_median
    print("\nRiepilogo rel_L1_median:")
    print(f"- RNADE:                {rnade_rel_L1_median:.3e}")
    print(f"- RealNVP:              {realnvp_rel_L1_median:.3e}")
    print(f"- KDE:                  {kde_rel_L1_median:.3e}")
    print(f"- GMM:                  {gmm_rel_L1_median:.3e}\n")



    plot_density_comparisons_filter(
        estimates=[
            ("RNADE", est_rnade),
            ("RealNVP", est_realnvp),
            ("KDE", est_kde),
            ("GMM", est_gmm)
        ],
        dim=D,
        true_density=true_density,
        out_dir="results",
        dataset="exp"
    )
    # Salvataggio risultati e permutazione nel CSV
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results_10_dim_exp_5000.csv')

    # Prepara stringa permutazione (es. "0 3 1 2 ...")
    perm_str = ' '.join(str(i) for i in perm)

    # Salva metriche + permutazione
    results = {
        'seed': GLOBAL_SEED,
        'perm': perm_str,
        'gmm_rmse': gmm_rmse,
        'kde_rmse': kde_rmse,
        'rnade_rmse': rnade_rmse,
        'realnvp_rmse': realnvp_rmse,
        'gmm_ise': gmm_ise,
        'kde_ise': kde_ise,
        'rnade_ise': rnade_ise,
        'realnvp_ise': realnvp_ise,
        'gmm_rel_L1': gmm_rel_L1,
        'kde_rel_L1': kde_rel_L1,
        'rnade_rel_L1': rnade_rel_L1,
        'realnvp_rel_L1': realnvp_rel_L1,
        'gmm_rel_L1_median': gmm_rel_L1_median,
        'kde_rel_L1_median': kde_rel_L1_median,
        'rnade_rel_L1_median': rnade_rel_L1_median,
        'realnvp_rel_L1_median': realnvp_rel_L1_median,
        'gmm_rmse_median': gmm_rmse_median,
        'kde_rmse_median': kde_rmse_median,
        'rnade_rmse_median': rnade_rmse_median,
        'realnvp_rmse_median': realnvp_rmse_median,
        'gmm_ise_median': gmm_ise_median,
        'kde_ise_median': kde_ise_median,
        'rnade_ise_median': rnade_ise_median,
        'realnvp_ise_median': realnvp_ise_median
    }
    df_results = pd.DataFrame([results])
    if os.path.exists(results_path):
        df_results.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df_results.to_csv(results_path, index=False)
        print(f"Risultati salvati in {results_path}:")
        print(df_results)



if __name__ == "__main__":
    main()



