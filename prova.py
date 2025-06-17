GLOBAL_SEED = 5

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
    evaluate_normal,evaluate_rnade,evaluate_realnvp,fit_kde_bandwidth_train_val,evaluate_kde_rmse,
    fit_gmm,fit_gmm_train_val,evaluate_gmm_rmse)


#!/usr/bin/env python3


# Generazione dati senza x3, x9 e x10
# Le altre variabili rinominate in successione: x1..x7
def generate_data(N, seed=None):
    """
    Genera un ndarray (N,13) con colonne x1..x13.
    """
    if seed is not None:
        np.random.seed(seed)

    # x1..x7 come prima
    x1 = np.random.randn(N)
    mu2 = np.sin(2 * x1)
    sd2 = 0.5 + 0.1 * np.abs(np.sin(x1))
    x2 = np.random.normal(mu2, sd2)

    mu3 = np.cos(x1)
    sd3 = 0.3 + 0.05 * np.abs(np.cos(x1))
    x3 = np.random.normal(mu3, sd3)

    mu4 = x1 * np.sin(x1)
    sd4 = 0.5 + 0.2 * np.abs(x1)
    x4 = np.random.normal(mu4, sd4)

    mu5 = np.tanh(x1)
    sd5 = 0.4 + 0.1 * x1**2
    x5 = np.random.normal(mu5, sd5)

    mu6 = np.log(np.abs(x1) + 1)
    sd6 = 0.3 + 0.1 * np.abs(x1)
    x6 = np.random.normal(mu6, sd6)

    mu7 = np.exp(-0.2 * x1)
    sd7 = 0.6
    x7 = np.random.normal(mu7, sd7)

    # x8 ~ Laplace(0, b) con varianza = 1 => b = 1/sqrt(2)
    b8 = 1 / np.sqrt(2)
    x8 = np.random.laplace(0, b8, size=N)

    # x9 e x10 come prima, condizionate su x8
    mu9 = np.sin(2 * x8)
    sd9 = 0.5 + 0.1 * np.abs(np.sin(x8))
    x9 = np.random.normal(mu9, sd9)

    mu10 = np.cos(x8)
    sd10 = 0.3 + 0.05 * np.abs(np.cos(x8))
    x10 = np.random.normal(mu10, sd10)

    # x11, x12: combinazioni lineari di x1 e x8 + rumore
    # var(x1+x8)=2 => (x1+x8)/sqrt(2) ha var=1
    mu11 = (x1 + x8) / np.sqrt(2)
    sd11 = 0.1
    x11 = np.random.normal(mu11, sd11)

    mu12 = (x1 - x8) / np.sqrt(2)
    sd12 = 0.1
    x12 = np.random.normal(mu12, sd12)

    # x13: prodotto x1*x8 ha var=var(x1)*var(x8)=1
    mu13 = x1 * x8
    sd13 = 0.1
    x13 = np.random.normal(mu13, sd13)

    return np.stack([x1, x2, x3, x4, x5, x6, x7,
                     x8, x9, x10, x11, x12, x13], axis=1)


# 2) Densità teorica congiunta

def pdf_x(X):
    """
    Calcola la densità congiunta per ogni riga di X (n,13).
    """
    X = np.asarray(X)
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13 = X.T

    # log-pdf di x1..x7 come prima
    lp = norm(0, 1).logpdf(x1)
    lp += norm(np.sin(2*x1), 0.5 + 0.1*np.abs(np.sin(x1))).logpdf(x2)
    lp += norm(np.cos(x1), 0.3 + 0.05*np.abs(np.cos(x1))).logpdf(x3)
    lp += norm(x1*np.sin(x1), 0.5 + 0.2*np.abs(x1)).logpdf(x4)
    lp += norm(np.tanh(x1), 0.4 + 0.1*x1**2).logpdf(x5)
    lp += norm(np.log(np.abs(x1)+1), 0.3 + 0.1*np.abs(x1)).logpdf(x6)
    lp += norm(np.exp(-0.2*x1), 0.6).logpdf(x7)

    # log-pdf di x8
    b8 = 1/np.sqrt(2)
    lp += laplace(0, b8).logpdf(x8)

    # log-pdf di x9, x10 condizionate su x8
    lp += norm(np.sin(2*x8), 0.5 + 0.1*np.abs(np.sin(x8))).logpdf(x9)
    lp += norm(np.cos(x8), 0.3 + 0.05*np.abs(np.cos(x8))).logpdf(x10)

    # log-pdf di x11, x12, x13 condizionate su x1,x8
    lp += norm((x1+x8)/np.sqrt(2), 0.1).logpdf(x11)
    lp += norm((x1-x8)/np.sqrt(2), 0.1).logpdf(x12)
    lp += norm(x1*x8, 0.1).logpdf(x13)

    return np.exp(lp)



import copy
import torch
from torch import optim
import math
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture


'''
def main():
    seed=1234
    data=generate_data(10000,seed=seed)
    random_state = seed
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=random_state)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)
    # Calcolo della media per ogni colonna
    mean_per_feature = np.mean(train_data, axis=0)

    # Calcolo della varianza per ogni colonna
    variance_per_feature = np.var(train_data, axis=0)
    true_density = pdf_x(test_data)
    gmm_model,best_n = fit_gmm_cv(train_data)
    gmm_rmse, gmm_density = evaluate_gmm_rmse(test_data, gmm_model, true_density)
    print(gmm_rmse)
    kde_model,bw = fit_kde_bandwidth_cv(train_data)
    kde_rmse, kde_density = evaluate_kde_rmse(test_data, kde_model, true_density)
    print(kde_rmse)

    plot_scatter_matrix(data)

    # Stampa dei risultati
    for i, (mean, var) in enumerate(zip(mean_per_feature, variance_per_feature)):
        print(f"Feature {i}: media = {mean:.4f}, varianza = {var:.4f}")
    
    #train a RNADE model with early stopping:
    hyperparams_grid = {
    "init_lr": [0.025,0.0125],
    "weight_decay": [0.0,0.001],
    "num_components": [1,2,5,10]
    }
    hyperparams_grid = {
    "init_lr": [0.0125],
    "weight_decay": [0.001],
    "num_components": [5]
    }
    
    
    hidden_units = 50
    best_params, best_val_ll = grid_search(train_data, valid_data, hidden_units, hyperparams_grid)
    print("Migliori iperparametri (Modello 1):", best_params)
    print("Miglior Val LL (Modello 1):", best_val_ll)
    rnade=RNADE(train_data.shape[1], hidden_units,best_params['num_components']) #aumenta patience e epochs se serve
    rnade = train_rnade_finale(rnade, train_data, best_cv_val_ll=best_val_ll,
                                num_epochs=4000, batch_size=500,
                                init_lr=best_params["init_lr"], weight_decay=best_params["weight_decay"])


    #rnade_rmse=evaluate_rnade(test_data, rnade, pdf_x)
    #print(rnade_rmse)
    
    # 2) costruisci DataLoader per train
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    val_tensor = torch.tensor(valid_data, dtype=torch.float32)
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    # 3) istanzia e allena RealNVP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realnvp = RealNVP(
        n_blocks=5,
        input_size=train_data.shape[1],
        hidden_size=128,
        n_hidden=2,
        batch_norm=True
    )
    realnvp = train_realnvp(realnvp, train_loader,val_loader=val_loader, lr=1e-3, epochs=40, device=device,patience=5)
    
    
    # Valutazione modelli
    rnade_rmse, rnade_density = evaluate_rnade(test_data, rnade, true_density)
    realnvp_rmse, realnvp_density = evaluate_realnvp(test_data, realnvp, true_density)
    
    # Stima e valutazione normale multivariata
    mean, cov = fit_multivariate_normal(train_data)
    mvn_rmse, mvn_density = evaluate_normal(test_data, mean, cov, true_density)
    

    # Preparazione dati per plotting
    sorted_idx = np.argsort(true_density)
    true_sorted = true_density[sorted_idx]
    rnade_sorted = rnade_density[sorted_idx]
    realnvp_sorted = realnvp_density[sorted_idx]
    mvn_sorted = mvn_density[sorted_idx]
    kde_sorted = kde_density[sorted_idx]
    gmm_sorted = gmm_density[sorted_idx]
    

    # Creazione plot combinato
    plt.figure(figsize=(12, 7))
    plt.plot(true_sorted, label='Teorica', linestyle=':', lw=3, alpha=0.9)
    plt.plot(rnade_sorted, label=f'RNADE ({rnade_rmse:.2e})', alpha=0.8)
    plt.plot(realnvp_sorted, label=f'RealNVP ({realnvp_rmse:.2e})', alpha=0.8)
    plt.plot(mvn_sorted, label=f'Normale ({mvn_rmse:.2e})', alpha=0.8)
    plt.plot(kde_sorted, label=f'KDE ({kde_rmse:.2e})')
    plt.plot(gmm_sorted, label=f'GMM ({gmm_rmse:.2e})')
    plt.plot(true_sorted, label='Teorica', linestyle=':', lw=3, alpha=0.9)
    
    plt.xlabel('Campioni ordinati per densità teorica', fontsize=12)
    plt.ylabel('Densità', fontsize=12)
    plt.title('Confronto Metodi di Stima della Densità', fontsize=14)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.2)
    #plt.yscale('log')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/density_comparison_all_methods2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Stampa riepilogo
    print("\nRiepilogo RMSE:")
    print(f"- RNADE: {rnade_rmse:.3e}")
    print(f"- RealNVP: {realnvp_rmse:.3e}")
    print(f"- Normale Multivariata: {mvn_rmse:.3e}")
    print(f"- KDE: {kde_rmse:.3e}")
    print(f"- GMM: {gmm_rmse:.3e}\n")

if __name__ == "__main__":
    main()

'''



# Definisci un seed globale e worker_init_fn a livello di modulo
#GLOBAL_SEED = 123

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


'''
# --- Main ---
def main():
    # Imposta seed globale
    set_seed(GLOBAL_SEED)

    # Generazione dati
    data = generate_data(10000)

    # Split train/valid/test
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=GLOBAL_SEED)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=GLOBAL_SEED)
    

    # Calcolo densità vera
    true_density = pdf_x(test_data)

    # --- GMM ---
    set_seed(GLOBAL_SEED)
    gmm_model, best_n = fit_gmm_train_val(train_data, valid_data,random_state=GLOBAL_SEED)
    gmm_rmse, gmm_density = evaluate_gmm_rmse(test_data,gmm_model, true_density)

    # --- KDE ---
    set_seed(GLOBAL_SEED)
    kde_model, bw = fit_kde_bandwidth_train_val(train_data, valid_data,random_state=GLOBAL_SEED)
    kde_rmse, kde_density = evaluate_kde_rmse(test_data, kde_model, true_density)

    # Plot scatter matrix
    plot_scatter_matrix(data)

    # Statistiche descrittive
    set_seed(GLOBAL_SEED)
    mean_per_feature = np.mean(train_data, axis=0)
    variance_per_feature = np.var(train_data, axis=0)
    for i, (mean_f, var_f) in enumerate(zip(mean_per_feature, variance_per_feature)):
        print(f"Feature {i+1}: media = {mean_f:.4f}, varianza = {var_f:.4f}")

    # --- RNADE Grid Search e Training ---
    set_seed(GLOBAL_SEED)
    
    hidden_units = 60
    
    rnade = RNADE(train_data.shape[1], hidden_units, 5)
    rnade = train_rnade(rnade, train_data, valid_data, num_epochs=1000, batch_size=256, init_lr=0.025, weight_decay=0.0001,patience=40)

    # --- DataLoader per PyTorch ---
    set_seed(GLOBAL_SEED)
    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256,
                              shuffle=True, generator=generator,
                              worker_init_fn=worker_init_fn, num_workers=4)

    set_seed(GLOBAL_SEED)
    generator_val = torch.Generator()
    generator_val.manual_seed(GLOBAL_SEED)
    val_tensor = torch.tensor(valid_data, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=256,
                            shuffle=True, generator=generator_val,
                            worker_init_fn=worker_init_fn, num_workers=4)

    # --- RealNVP ---
    set_seed(GLOBAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realnvp = RealNVP(n_blocks=10, input_size=train_data.shape[1],
                      hidden_size=128, n_hidden=2, batch_norm=True)
    realnvp = train_realnvp(realnvp, train_loader, val_loader=val_loader,
                             lr=1e-3, epochs=40,
                             device=device, patience=5, min_delta=1e-4)

    # --- Valutazioni finali ---
    set_seed(GLOBAL_SEED)
    rnade_rmse, rnade_density = evaluate_rnade(test_data, rnade, true_density)
    set_seed(GLOBAL_SEED)
    realnvp_rmse, realnvp_density = evaluate_realnvp(test_data, realnvp, true_density)
    set_seed(GLOBAL_SEED)
    mean_mvn, cov_mvn = fit_multivariate_normal(train_data)
    mvn_rmse, mvn_density = evaluate_normal(test_data, mean_mvn, cov_mvn, true_density)

    # Stampa riepilogo
    print("\nRiepilogo RMSE:")
    print(f"- RNADE: {rnade_rmse:.3e}")
    print(f"- RealNVP: {realnvp_rmse:.3e}")
    print(f"- Normale Multivariata: {mvn_rmse:.3e}")
    print(f"- KDE: {kde_rmse:.3e}")
    print(f"- GMM: {gmm_rmse:.3e}\n")
    
    # --- Salvataggio risultati in CSV nella cartella results ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results_13_dim.csv')
    results = {
        'seed': GLOBAL_SEED,
        'gmm_rmse': gmm_rmse,
        'kde_rmse': kde_rmse,
        'rnade_rmse': rnade_rmse,
        'realnvp_rmse': realnvp_rmse,
        'mvn_rmse': mvn_rmse
    }
    df_results = pd.DataFrame([results])
    # Append if exists, else write header
    if os.path.exists(results_path):
        df_results.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df_results.to_csv(results_path, index=False)
        print(f"Risultati salvati in {results_path}:")
        print(df_results)
'''


def main():
    # Imposta seed globale
    set_seed(GLOBAL_SEED)

    # Generazione dati (shape: [n_samples, n_features])
    data = generate_data(10000)

    # Split train/valid/test
    train_data, temp_data = train_test_split(
        data, test_size=0.4, random_state=GLOBAL_SEED
    )
    valid_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=GLOBAL_SEED
    )

    # Calcolo densità vera sul TEST prima dello shuffle delle colonne
    true_density = pdf_x(test_data)

    # Genera permutazione delle colonne e shuffle dei dataset
    rng = np.random.RandomState(GLOBAL_SEED)
    n_features = train_data.shape[1]
    perm = rng.permutation(n_features)
    train_data = train_data[:, perm]
    valid_data = valid_data[:, perm]
    test_data  = test_data[:,  perm]

    # --- GMM ---
    set_seed(GLOBAL_SEED)
    gmm_model, best_n = fit_gmm_train_val(
        train_data, valid_data,
        n_components_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
        random_state=GLOBAL_SEED
    )
    gmm_rmse, _ = evaluate_gmm_rmse(
        test_data, gmm_model, true_density
    )

    # --- KDE ---
    set_seed(GLOBAL_SEED)
    kde_model, bw = fit_kde_bandwidth_train_val(
        train_data, valid_data, random_state=GLOBAL_SEED
    )
    kde_rmse, _ = evaluate_kde_rmse(
        test_data, kde_model, true_density
    )

    # Plot scatter matrix sui dati originali
    plot_scatter_matrix(data)

    # Statistiche descrittive sul TRAIN permutato
    set_seed(GLOBAL_SEED)
    mean_per_feature = np.mean(train_data, axis=0)
    variance_per_feature = np.var(train_data, axis=0)
    for i, (mean_f, var_f) in enumerate(zip(mean_per_feature, variance_per_feature)):
        print(f"Feature {i+1}: media = {mean_f:.4f}, varianza = {var_f:.4f}")

    # --- RNADE ---
    set_seed(GLOBAL_SEED)
    hidden_units = 100
    rnade = RNADE(train_data.shape[1], hidden_units, 5)
    rnade = train_rnade(
        rnade, train_data, valid_data,
        num_epochs=2000, batch_size=256,
        init_lr=0.02, weight_decay=0,
        patience=50
    )

    

    # DataLoader per PyTorch
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

    # --- RealNVP ---
    set_seed(GLOBAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realnvp = RealNVP(
        n_blocks=15,
        input_size=train_data.shape[1],
        hidden_size=128,
        n_hidden=2,
        batch_norm=True
    )
    realnvp = train_realnvp(
        realnvp, train_loader, val_loader=val_loader,
        lr=1e-3, epochs=60,
        device=device, patience=5, min_delta=1e-4
    )

    # --- Valutazioni finali su dati permutati ---
    set_seed(GLOBAL_SEED)
    rnade_rmse, _ = evaluate_rnade(
        test_data, rnade, true_density
    )
    set_seed(GLOBAL_SEED)
    realnvp_rmse, _ = evaluate_realnvp(
        test_data, realnvp, true_density
    )
    set_seed(GLOBAL_SEED)
    mean_mvn, cov_mvn = fit_multivariate_normal(train_data)
    mvn_rmse, _ = evaluate_normal(
        test_data, mean_mvn, cov_mvn, true_density
    )

    # Riepilogo RMSE
    print("\nRiepilogo RMSE:")
    print(f"- RNADE:                {rnade_rmse:.3e}")
    print(f"- RealNVP:              {realnvp_rmse:.3e}")
    print(f"- Normale Multivariata: {mvn_rmse:.3e}")
    print(f"- KDE:                  {kde_rmse:.3e}")
    print(f"- GMM:                  {gmm_rmse:.3e}\n")

    # Salvataggio risultati e permutazione nel CSV
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results_13_dim.csv')

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
        'mvn_rmse': mvn_rmse
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
 



 '''
# --- Main ---
def main():
    # Imposta seed globale
    set_seed(GLOBAL_SEED)

    # Generazione dati
    data = generate_data(10000)

    # Split train/valid/test
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=GLOBAL_SEED)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=GLOBAL_SEED)
    

    # Calcolo densità vera
    true_density = pdf_x(test_data)

    # --- GMM ---
    set_seed(GLOBAL_SEED)
    gmm_model, best_n = fit_gmm_train_val(train_data, valid_data,random_state=GLOBAL_SEED)
    gmm_rmse, gmm_density = evaluate_gmm_rmse(test_data,gmm_model, true_density)

    # --- KDE ---
    set_seed(GLOBAL_SEED)
    kde_model, bw = fit_kde_bandwidth_train_val(train_data, valid_data,random_state=GLOBAL_SEED)
    kde_rmse, kde_density = evaluate_kde_rmse(test_data, kde_model, true_density)

    # Plot scatter matrix
    plot_scatter_matrix(data)

    # Statistiche descrittive
    set_seed(GLOBAL_SEED)
    mean_per_feature = np.mean(train_data, axis=0)
    variance_per_feature = np.var(train_data, axis=0)
    for i, (mean_f, var_f) in enumerate(zip(mean_per_feature, variance_per_feature)):
        print(f"Feature {i+1}: media = {mean_f:.4f}, varianza = {var_f:.4f}")

    # --- RNADE Grid Search e Training ---
    set_seed(GLOBAL_SEED)
    
    hidden_units = 60
    
    rnade = RNADE(train_data.shape[1], hidden_units, 5)
    rnade = train_rnade(rnade, train_data, valid_data, num_epochs=1000, batch_size=256, init_lr=0.025, weight_decay=0.0001,patience=40)

    # --- DataLoader per PyTorch ---
    set_seed(GLOBAL_SEED)
    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256,
                              shuffle=True, generator=generator,
                              worker_init_fn=worker_init_fn, num_workers=4)

    set_seed(GLOBAL_SEED)
    generator_val = torch.Generator()
    generator_val.manual_seed(GLOBAL_SEED)
    val_tensor = torch.tensor(valid_data, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=256,
                            shuffle=True, generator=generator_val,
                            worker_init_fn=worker_init_fn, num_workers=4)

    # --- RealNVP ---
    set_seed(GLOBAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realnvp = RealNVP(n_blocks=10, input_size=train_data.shape[1],
                      hidden_size=128, n_hidden=2, batch_norm=True)
    realnvp = train_realnvp(realnvp, train_loader, val_loader=val_loader,
                             lr=1e-3, epochs=40,
                             device=device, patience=5, min_delta=1e-4)

    # --- Valutazioni finali ---
    set_seed(GLOBAL_SEED)
    rnade_rmse, rnade_density = evaluate_rnade(test_data, rnade, true_density)
    set_seed(GLOBAL_SEED)
    realnvp_rmse, realnvp_density = evaluate_realnvp(test_data, realnvp, true_density)
    set_seed(GLOBAL_SEED)
    mean_mvn, cov_mvn = fit_multivariate_normal(train_data)
    mvn_rmse, mvn_density = evaluate_normal(test_data, mean_mvn, cov_mvn, true_density)

    # Stampa riepilogo
    print("\nRiepilogo RMSE:")
    print(f"- RNADE: {rnade_rmse:.3e}")
    print(f"- RealNVP: {realnvp_rmse:.3e}")
    print(f"- Normale Multivariata: {mvn_rmse:.3e}")
    print(f"- KDE: {kde_rmse:.3e}")
    print(f"- GMM: {gmm_rmse:.3e}\n")
    
    # --- Salvataggio risultati in CSV nella cartella results ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results_13_dim.csv')
    results = {
        'seed': GLOBAL_SEED,
        'gmm_rmse': gmm_rmse,
        'kde_rmse': kde_rmse,
        'rnade_rmse': rnade_rmse,
        'realnvp_rmse': realnvp_rmse,
        'mvn_rmse': mvn_rmse
    }
    df_results = pd.DataFrame([results])
    # Append if exists, else write header
    if os.path.exists(results_path):
        df_results.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df_results.to_csv(results_path, index=False)
        print(f"Risultati salvati in {results_path}:")
        print(df_results)
'''



'''
def main():
    seed=1234
    data=generate_data(10000,seed=seed)
    random_state = seed
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=random_state)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)
    # Calcolo della media per ogni colonna
    mean_per_feature = np.mean(train_data, axis=0)

    # Calcolo della varianza per ogni colonna
    variance_per_feature = np.var(train_data, axis=0)
    true_density = pdf_x(test_data)
    gmm_model,best_n = fit_gmm_cv(train_data)
    gmm_rmse, gmm_density = evaluate_gmm_rmse(test_data, gmm_model, true_density)
    print(gmm_rmse)
    kde_model,bw = fit_kde_bandwidth_cv(train_data)
    kde_rmse, kde_density = evaluate_kde_rmse(test_data, kde_model, true_density)
    print(kde_rmse)

    plot_scatter_matrix(data)

    # Stampa dei risultati
    for i, (mean, var) in enumerate(zip(mean_per_feature, variance_per_feature)):
        print(f"Feature {i}: media = {mean:.4f}, varianza = {var:.4f}")
    
    #train a RNADE model with early stopping:
    hyperparams_grid = {
    "init_lr": [0.025,0.0125],
    "weight_decay": [0.0,0.001],
    "num_components": [1,2,5,10]
    }
    hyperparams_grid = {
    "init_lr": [0.0125],
    "weight_decay": [0.001],
    "num_components": [5]
    }
    
    
    hidden_units = 50
    best_params, best_val_ll = grid_search(train_data, valid_data, hidden_units, hyperparams_grid)
    print("Migliori iperparametri (Modello 1):", best_params)
    print("Miglior Val LL (Modello 1):", best_val_ll)
    rnade=RNADE(train_data.shape[1], hidden_units,best_params['num_components']) #aumenta patience e epochs se serve
    rnade = train_rnade_finale(rnade, train_data, best_cv_val_ll=best_val_ll,
                                num_epochs=4000, batch_size=500,
                                init_lr=best_params["init_lr"], weight_decay=best_params["weight_decay"])


    #rnade_rmse=evaluate_rnade(test_data, rnade, pdf_x)
    #print(rnade_rmse)
    
    # 2) costruisci DataLoader per train
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    val_tensor = torch.tensor(valid_data, dtype=torch.float32)
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    # 3) istanzia e allena RealNVP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realnvp = RealNVP(
        n_blocks=5,
        input_size=train_data.shape[1],
        hidden_size=128,
        n_hidden=2,
        batch_norm=True
    )
    realnvp = train_realnvp(realnvp, train_loader,val_loader=val_loader, lr=1e-3, epochs=40, device=device,patience=5)
    
    
    # Valutazione modelli
    rnade_rmse, rnade_density = evaluate_rnade(test_data, rnade, true_density)
    realnvp_rmse, realnvp_density = evaluate_realnvp(test_data, realnvp, true_density)
    
    # Stima e valutazione normale multivariata
    mean, cov = fit_multivariate_normal(train_data)
    mvn_rmse, mvn_density = evaluate_normal(test_data, mean, cov, true_density)
    

    # Preparazione dati per plotting
    sorted_idx = np.argsort(true_density)
    true_sorted = true_density[sorted_idx]
    rnade_sorted = rnade_density[sorted_idx]
    realnvp_sorted = realnvp_density[sorted_idx]
    mvn_sorted = mvn_density[sorted_idx]
    kde_sorted = kde_density[sorted_idx]
    gmm_sorted = gmm_density[sorted_idx]
    

    # Creazione plot combinato
    plt.figure(figsize=(12, 7))
    plt.plot(true_sorted, label='Teorica', linestyle=':', lw=3, alpha=0.9)
    plt.plot(rnade_sorted, label=f'RNADE ({rnade_rmse:.2e})', alpha=0.8)
    plt.plot(realnvp_sorted, label=f'RealNVP ({realnvp_rmse:.2e})', alpha=0.8)
    plt.plot(mvn_sorted, label=f'Normale ({mvn_rmse:.2e})', alpha=0.8)
    plt.plot(kde_sorted, label=f'KDE ({kde_rmse:.2e})')
    plt.plot(gmm_sorted, label=f'GMM ({gmm_rmse:.2e})')
    plt.plot(true_sorted, label='Teorica', linestyle=':', lw=3, alpha=0.9)
    
    plt.xlabel('Campioni ordinati per densità teorica', fontsize=12)
    plt.ylabel('Densità', fontsize=12)
    plt.title('Confronto Metodi di Stima della Densità', fontsize=14)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.2)
    #plt.yscale('log')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/density_comparison_all_methods2.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Stampa riepilogo
    print("\nRiepilogo RMSE:")
    print(f"- RNADE: {rnade_rmse:.3e}")
    print(f"- RealNVP: {realnvp_rmse:.3e}")
    print(f"- Normale Multivariata: {mvn_rmse:.3e}")
    print(f"- KDE: {kde_rmse:.3e}")
    print(f"- GMM: {gmm_rmse:.3e}\n")

if __name__ == "__main__":
    main()

'''
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



GLOBAL_SEED = 578221

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
    evaluate_normal,evaluate_rnade,evaluate_realnvp,fit_kde_bandwidth_train_val,evaluate_kde_rmse,
    fit_gmm,fit_gmm_train_val,evaluate_gmm_rmse,plot_density_comparisons,evaluate_gmm_ise,evaluate_kde_ise,evaluate_rnade_ise,
    evaluate_realnvp_ise)

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
    Genera N campioni da una normale multivariata con media e matrice di covarianza sigma.

    Parametri
    ----------
    N     : int
        Numero di campioni da generare.
    seed  : int
        Seed per il generatore di numeri casuali.
    sigma : np.ndarray, shape (d, d)
        Matrice di covarianza (deve essere definita positiva).
    mean  : np.ndarray, shape (d,), facoltativo
        Vettore di medie; default vettore di zeri.

    Ritorna
    -------
    X : np.ndarray, shape (N, d)
        Matrice dei campioni generati.
    """
    # Imposta la seed per riproducibilità
    rng = np.random.default_rng(seed)
    # Dimensione dal sigma
    d = sigma.shape[0]
    # Se mean non specificata, vettore di zeri
    if mean is None:
        mean = np.zeros(d)
    # Genera campioni multivariati
    X = rng.multivariate_normal(mean, sigma, size=N)
    return X


def mvn_density(X: np.ndarray,
                sigma: np.ndarray,
                mean: np.ndarray = None) -> np.ndarray:
    """
    Calcola la densità di punti X sotto una normale multivariata con
    media e sigma dati usando scipy.stats.multivariate_normal.

    Parametri
    ----------
    X     : np.ndarray, shape (n_samples, d)
        Punti in cui valutare la densità.
    sigma : np.ndarray, shape (d, d)
        Matrice di covarianza.
    mean  : np.ndarray, shape (d,), facoltativo
        Vettore di medie; default vettore di zeri.

    Ritorna
    -------
    densities : np.ndarray, shape (n_samples,)
        Valori di densità per ciascun punto.
    """
    # Media predefinita
    if mean is None:
        mean = np.zeros(sigma.shape[0])
    
    # Usa scipy per calcolare la densità
    mvn = multivariate_normal(mean=mean, cov=sigma)
    densities = mvn.pdf(X)
    
    return densities
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
    N = 100000
    D=10
    sigma = ar1_cor(D, 0.9)
    data = generate_data(N, GLOBAL_SEED, sigma)
    
    # Split train/valid/test
    train_data, temp_data = train_test_split(
        data, test_size=0.9, random_state=GLOBAL_SEED
    )
    valid_data, test_data = train_test_split(
        temp_data, test_size=44/45, random_state=GLOBAL_SEED
    )

    print(f"Dimensione del set di addestramento: {train_data.shape}")
    print(f"Dimensione del set di validazione: {valid_data.shape}")
    print(f"Dimensione del set di test: {test_data.shape}")

    # Calcolo densità vera sul TEST prima dello shuffle delle colonne
    true_density = mvn_density(test_data, sigma)

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
        n_components_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
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
    rnade = RNADE(train_data.shape[1], hidden_units, 5)
    rnade = train_rnade(
        rnade, train_data, valid_data,
        num_epochs=2000, batch_size=256,
        init_lr=0.02, weight_decay=0,
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

    # --- Valutazioni finali su dati permutati ---
    #set_seed(GLOBAL_SEED)
    rnade_rmse, est_rnade = evaluate_rnade(
        test_data, rnade, true_density
    )
    #set_seed(GLOBAL_SEED)
    realnvp_rmse, est_realnvp = evaluate_realnvp(
        test_data, realnvp, true_density
    )
    kde_rmse, est_kde = evaluate_kde_rmse(
        test_data, kde_model, true_density
    )
    gmm_rmse, est_gmm = evaluate_gmm_rmse(
        test_data, gmm_model, true_density
    )


    # Riepilogo RMSE
    print("\nRiepilogo RMSE:")
    print(f"- RNADE:                {rnade_rmse:.3e}")
    print(f"- RealNVP:              {realnvp_rmse:.3e}")
    print(f"- KDE:                  {kde_rmse:.3e}")
    print(f"- GMM:                  {gmm_rmse:.3e}\n")

    #Calcolo ISE
    rnade_ise, _ = evaluate_rnade_ise(
        test_data, rnade, true_density
    )
    realnvp_ise, _ = evaluate_realnvp_ise(
        test_data, realnvp, true_density
    )
    kde_ise, _ = evaluate_kde_ise(
        test_data, kde_model, true_density
    )
    gmm_ise, _ = evaluate_gmm_ise(
        test_data, gmm_model, true_density
    )

    # Riepilogo ISE
    print("\nRiepilogo ISE:")
    print(f"- RNADE:                {rnade_ise:.3e}")
    print(f"- RealNVP:              {realnvp_ise:.3e}")
    print(f"- KDE:                  {kde_ise:.3e}")
    print(f"- GMM:                  {gmm_ise:.3e}\n")
   

    plot_density_comparisons(
        estimates=[
            ("RNDE", est_rnade),
            ("RealNVP", est_realnvp),
            ("KDE", est_kde),
            ("GMM", est_gmm)
        ],
        dim=D,
        true_density=true_density,
        out_dir="results"
    )
    # Salvataggio risultati e permutazione nel CSV
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results_10_dim_mnvtnorm.csv')

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
        'realnvp_ise': realnvp_ise
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



    GLOBAL_SEED = 5

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
    evaluate_normal,evaluate_rnade,evaluate_realnvp,fit_kde_bandwidth_train_val,evaluate_kde_rmse,
    fit_gmm,fit_gmm_train_val,evaluate_gmm_rmse,plot_density_comparisons,evaluate_gmm_ise,evaluate_kde_ise,evaluate_rnade_ise,
    evaluate_realnvp_ise)

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
    Genera N campioni da una normale multivariata con media e matrice di covarianza sigma.

    Parametri
    ----------
    N     : int
        Numero di campioni da generare.
    seed  : int
        Seed per il generatore di numeri casuali.
    sigma : np.ndarray, shape (d, d)
        Matrice di covarianza (deve essere definita positiva).
    mean  : np.ndarray, shape (d,), facoltativo
        Vettore di medie; default vettore di zeri.

    Ritorna
    -------
    X : np.ndarray, shape (N, d)
        Matrice dei campioni generati.
    """
    # Imposta la seed per riproducibilità
    rng = np.random.default_rng(seed)
    # Dimensione dal sigma
    d = sigma.shape[0]
    # Se mean non specificata, vettore di zeri
    if mean is None:
        mean = np.zeros(d)
    # Genera campioni multivariati
    X = rng.multivariate_normal(mean, sigma, size=N)
    return X


def mvn_density(X: np.ndarray,
                sigma: np.ndarray,
                mean: np.ndarray = None) -> np.ndarray:
    """
    Calcola la densità di punti X sotto una normale multivariata con
    media e sigma dati usando scipy.stats.multivariate_normal.

    Parametri
    ----------
    X     : np.ndarray, shape (n_samples, d)
        Punti in cui valutare la densità.
    sigma : np.ndarray, shape (d, d)
        Matrice di covarianza.
    mean  : np.ndarray, shape (d,), facoltativo
        Vettore di medie; default vettore di zeri.

    Ritorna
    -------
    densities : np.ndarray, shape (n_samples,)
        Valori di densità per ciascun punto.
    """
    # Media predefinita
    if mean is None:
        mean = np.zeros(sigma.shape[0])
    
    # Usa scipy per calcolare la densità
    mvn = multivariate_normal(mean=mean, cov=sigma)
    densities = mvn.pdf(X)
    
    return densities
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
    N = 30000
    D=20
    sigma = ar1_cor(D, 0.9)
    data = generate_data(N, GLOBAL_SEED, sigma)
    
    # Split train/valid/test
    train_data, temp_data = train_test_split(
        data, test_size=22/30, random_state=GLOBAL_SEED
    )
    valid_data, test_data = train_test_split(
        temp_data, test_size=10/11, random_state=GLOBAL_SEED
    )

    print(f"Dimensione del set di addestramento: {train_data.shape}")
    print(f"Dimensione del set di validazione: {valid_data.shape}")
    print(f"Dimensione del set di test: {test_data.shape}")

    # Calcolo densità vera sul TEST prima dello shuffle delle colonne
    true_density = mvn_density(test_data, sigma)
    plot_scatter_matrix(test_data, save_path='results/test_scatter_matrix_20_d_mvtnorm.png')

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
        n_components_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
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
    rnade = RNADE(train_data.shape[1], hidden_units, 5)
    rnade = train_rnade(
        rnade, train_data, valid_data,
        num_epochs=2000, batch_size=256,
        init_lr=0.02, weight_decay=0,
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

    # --- Valutazioni finali su dati permutati ---
    #set_seed(GLOBAL_SEED)
    rnade_rmse, est_rnade = evaluate_rnade(
        test_data, rnade, true_density
    )
    #set_seed(GLOBAL_SEED)
    realnvp_rmse, est_realnvp = evaluate_realnvp(
        test_data, realnvp, true_density
    )
    kde_rmse, est_kde = evaluate_kde_rmse(
        test_data, kde_model, true_density
    )
    gmm_rmse, est_gmm = evaluate_gmm_rmse(
        test_data, gmm_model, true_density
    )


    # Riepilogo RMSE
    print("\nRiepilogo RMSE:")
    print(f"- RNADE:                {rnade_rmse:.3e}")
    print(f"- RealNVP:              {realnvp_rmse:.3e}")
    print(f"- KDE:                  {kde_rmse:.3e}")
    print(f"- GMM:                  {gmm_rmse:.3e}\n")

    #Calcolo ISE
    rnade_ise, _ = evaluate_rnade_ise(
        test_data, rnade, true_density
    )
    realnvp_ise, _ = evaluate_realnvp_ise(
        test_data, realnvp, true_density
    )
    kde_ise, _ = evaluate_kde_ise(
        test_data, kde_model, true_density
    )
    gmm_ise, _ = evaluate_gmm_ise(
        test_data, gmm_model, true_density
    )

    # Riepilogo ISE
    print("\nRiepilogo ISE:")
    print(f"- RNADE:                {rnade_ise:.3e}")
    print(f"- RealNVP:              {realnvp_ise:.3e}")
    print(f"- KDE:                  {kde_ise:.3e}")
    print(f"- GMM:                  {gmm_ise:.3e}\n")
   

    plot_density_comparisons(
        estimates=[
            ("RNDE", est_rnade),
            ("RealNVP", est_realnvp),
            ("KDE", est_kde),
            ("GMM", est_gmm)
        ],
        dim=D,
        true_density=true_density,
        out_dir="results"
    )
    # Salvataggio risultati e permutazione nel CSV
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'results_20_dim_mnvtnorm.csv')

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
        'realnvp_ise': realnvp_ise
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