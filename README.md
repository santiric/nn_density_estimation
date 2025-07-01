# Density Estimation Comparison

Questo progetto implementa e confronta quattro metodi di stima della densità:
- **Gaussian Mixture Models (GMM)**
- **Kernel Density Estimation (KDE)**
- **RNADE**: autoregressive neural density estimator.
- **RealNVP**: normalizing flow-based neural density estimator.

Lo studio valuta questi metodi su:
- **Due distribuzioni di dati**:
  - Scenario A: normale multivariata.
  - Scenario B: trasformata esponenziale dello scenario A (es. Y = exp(0.7 X)).
- **Tre dimensionalità**: 5D, 10D, 20D.
- **Due dimensioni del dataset**: 5000 e 10000 osservazioni.

---

## Installazione

Clonare il repository:
   ```bash
   git clone https://github.com/santiric/mn_density_estimation.git
  ```
Consiglio di creare un ambiente virtuale 
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
che contenga il codice disponibile in https://github.com/kamenbliznashki/normalizing_flows , che verrà usato dalle funzioni contenute in functions.py.


## Librerie necessarie

PyTorch: 2.2.5  
scikit-learn: 2.7.0+cpu  
SciPy: 1.6.1  
pandas: 1.15.2  
matplotlib: 2.2.3  
Python: 3.10.1  


