import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from itertools import product
import copy

class RNADE(nn.Module):
    def __init__(self, input_dim, hidden_units, num_components):
        """
        Inizializza il modello RNADE – Mixture of Gaussians.

        Parametri:
         - input_dim: numero di variabili (D).
         - hidden_units: numero di unità nella hidden layer (H). (Tipicamente H=50 per i dati low-dimensional come in Uria et al., 2016)
         - num_components: numero di componenti della miscela (C) per ogni condizionale.
        """
        super(RNADE, self).__init__() #Inizializza il modulo base di PyTorch, necessario per far funzionare il modello.
        self.input_dim = input_dim        # D
        self.hidden_units = hidden_units  # H
        self.num_components = num_components  # C

        # Parametri condivisi
        self.W = nn.Parameter(torch.randn(hidden_units, input_dim) * 0.01)  # H x D
        self.c = nn.Parameter(torch.zeros(hidden_units))                   # H

        # Parametri specifici per ogni dimensione
        #guarda il paper per capirci di più, stai inizializzando una lista di matrici H x C
        self.b_pi = nn.ParameterList([nn.Parameter(torch.zeros(num_components)) for _ in range(input_dim)])
        self.V_pi = nn.ParameterList([nn.Parameter(torch.randn(hidden_units, num_components) * 0.01)
                                       for _ in range(input_dim)])

        self.b_mu = nn.ParameterList([nn.Parameter(torch.zeros(num_components)) for _ in range(input_dim)])
        self.V_mu = nn.ParameterList([nn.Parameter(torch.randn(hidden_units, num_components) * 0.01)
                                       for _ in range(input_dim)])

        self.b_sigma = nn.ParameterList([nn.Parameter(torch.zeros(num_components)) for _ in range(input_dim)])
        self.V_sigma = nn.ParameterList([nn.Parameter(torch.randn(hidden_units, num_components) * 0.01)
                                          for _ in range(input_dim)])

    def forward(self, x):
        """
        Esegue il forward pass in modo autoregressivo.

        Parametri:
         - x: tensore di forma [batch_size, input_dim], ovvero [N,D]
        

        Restituisce:
         - log_prob: log-likelihood totale per ogni esempio.
        """
        batch_size = x.size(0) #N
        D = self.input_dim
        H = self.hidden_units
        C = self.num_components

        order = list(range(D)) #ordine utoregressivo

        # Stato iniziale della rete (bias nascosto)
        a = self.c.unsqueeze(0).expand(batch_size, H) #a1=c (eq.4), è una matrice con tanti c uguali affiancati, uno per ogni esempio,
        #così è più comodo farci algebra lineare sull'intero batch
        log_probs = []
        self.last_sigma = []  # Lista per salvare σ per ciascuna dimensione, da usare nel gradiente(vedi fine capitolo 3)

        # Processa in maniera autoregressiva ogni dimensione
        for d in range(D):
            current_index = order[d]
            h = F.relu(a)  # Attivazione nascosta con ReLU

            # Calcola i parametri della miscela per la dimensione corrente
            z_pi = self.b_pi[current_index] + h @ self.V_pi[current_index] #calcola per la dimensione d i vari z(c) in parallelo(eq.20)
            pi = F.softmax(z_pi, dim=1) #(17)

            z_mu = self.b_mu[current_index] + h @ self.V_mu[current_index]
            mu = z_mu  # Nessuna attivazione aggiuntiva per μ

            z_sigma = self.b_sigma[current_index] + h @ self.V_sigma[current_index]
            sigma = torch.exp(z_sigma)  # σ deve essere positivo
            self.last_sigma.append(sigma)  # Salva σ per l'aggiornamento dei gradienti

            # Calcola la densità della Gaussiana
            x_d = x[:, current_index].unsqueeze(1)  # [batch_size, 1]
            normalizer = 1.0 / (torch.sqrt(torch.tensor(2 * np.pi)) * sigma)
            exponent = -0.5 * ((x_d - mu) / sigma) ** 2
            gauss = normalizer * torch.exp(exponent)

            # Densità condizionale come somma pesata (mixing coefficient)
            p_cond = torch.sum(pi * gauss, dim=1) #(16)
            log_p_cond = torch.log(p_cond + 1e-10)  # Stabilità numerica
            log_probs.append(log_p_cond)

            # Aggiornamento autoregressivo dello stato nascosto
            a = a + x[:, current_index].unsqueeze(1) * self.W[:, current_index].unsqueeze(0) #(5) credo

        log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        return log_prob #ogni elemento è il log-likelihood totale per un esempio del batch
def train_rnade(model, train_data, valid_data, num_epochs=500, batch_size=100,
                init_lr=0.1, weight_decay=0.0, patience=20):
    """
    Addestra il modello RNADE utilizzando SGD sulla negativa log-likelihood.
    Viene applicato il weight decay solo al parametro condiviso W e,
    come suggerito in Uria et al. (2013), i gradienti dei parametri della media vengono scalati per σ.

    Parametri:
      - train_data: array NumPy [N_train, D] dei dati di training.
      - valid_data: array NumPy [N_valid, D] dei dati di validazione.
      - num_epochs: numero massimo di epoche (default 500).
      - batch_size: dimensione del minibatch (default 100).
      - init_lr: learning rate iniziale.
      - weight_decay: regolarizzazione per il parametro W.
      - patience: numero di epoche senza miglioramento per early stopping.

    Restituisce il modello con lo stato migliore in validazione.
    """

    '''
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    valid_tensor = torch.tensor(valid_data, dtype=torch.float32)
    '''
    # Assicura che train_data e valid_data siano convertiti correttamente in tensori senza warning
    if isinstance(train_data, torch.Tensor):
        train_tensor = train_data.clone().detach().float()
    else:
        train_tensor = torch.tensor(train_data, dtype=torch.float32)

    if isinstance(valid_data, torch.Tensor):
        valid_tensor = valid_data.clone().detach().float()
    else:
        valid_tensor = torch.tensor(valid_data, dtype=torch.float32)


    train_dataset = torch.utils.data.TensorDataset(train_tensor)#dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#dataloader
    #il dataloader si occupa di distribuire i dati in mini-batch

    # Applica weight decay solo al parametro condiviso W
    optimizer = optim.SGD([ #ottimizzatore SGD
        {'params': [model.W], 'weight_decay': weight_decay},
        {'params': [p for name, p in model.named_parameters() if name != 'W'], 'weight_decay': 0.0}
    ], lr=init_lr)

    best_val_ll = -float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 10  # utilizziamo 10 minibatch per epoca, come nelle specifiche
        for _ in range(num_batches):
            indices = np.random.choice(len(train_dataset), batch_size, replace=True)
            batch = train_tensor[indices]
            order = list(range(model.input_dim))  # ordine naturale(l'ho ridefinito in caso volessi implementare un apprendimento con un altro ordine)

            log_prob = model(batch)
            loss = -torch.mean(log_prob)

            optimizer.zero_grad()
            loss.backward()

            # Scaling del gradiente per i parametri della media:
            # Per ogni dimensione d, moltiplichiamo il gradiente relativo a b_mu e V_mu per la media (sul batch)
            # dei corrispondenti valori di σ, come indicato in Uria et al. (2013).
            for d in range(model.input_dim):
                # sigma_d ha forma [batch_size, num_components] per la dimensione d
                # Usiamo la media lungo il batch per ottenere uno scaling per ciascuna componente
                sigma_d = model.last_sigma[d].detach().mean(dim=0)  # shape: [num_components]
                if model.b_mu[d].grad is not None:
                    model.b_mu[d].grad.mul_(sigma_d)
                if model.V_mu[d].grad is not None:
                    # model.V_mu[d] ha forma [hidden_units, num_components] – il broadcasting con sigma_d.unsqueeze(0) è corretto
                    model.V_mu[d].grad.mul_(sigma_d.unsqueeze(0))

            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= (num_batches * batch_size)
        # Aggiorna il learning rate con decrescita lineare
        current_lr = init_lr * (1 - (epoch + 1) / num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.eval()
        with torch.no_grad():
            val_ll = model(valid_tensor).mean().item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val LL: {val_ll:.4f}, LR: {current_lr:.5f}")

        if val_ll > best_val_ll:
            best_val_ll = val_ll
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"Early stopping alla epoca {epoch+1} (miglioramento all'epoca {best_epoch+1})")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Miglior Val LL: {best_val_ll:.4f} all'epoca {best_epoch+1}")
    return model




def grid_search(train_data, valid_data, hidden_units, hyperparams_grid):
    """
    Esegue una grid search per trovare la migliore combinazione di iperparametri.

    Parametri:
      - train_data: array NumPy [N_train, D] dei dati di training.
      - valid_data: array NumPy [N_valid, D] dei dati di validazione.
      - input_dim: dimensione degli input (D).
      - hidden_units: numero di unità nella hidden layer (H).
      - hyperparams_grid: dizionario contenente liste di valori per:
            "init_lr": ad esempio, [0.1, 0.05, 0.025, 0.0125],
            "weight_decay": ad esempio, [0.0, 0.1, 0.01, 0.001],
            "num_components": ad esempio, [2, 5, 10, 20].

    Restituisce:
      - best_params: dizionario con la migliore combinazione di iperparametri.
      - best_val_ll: log-likelihood di validazione associato.
    """
    from itertools import product  # Se non già importato
    input_dim = train_data.shape[1]
    best_val_ll = -float('inf')
    best_params = None

    # Itera su tutte le combinazioni di iperparametri
    for init_lr, weight_decay, num_components in product(
            hyperparams_grid["init_lr"],
            hyperparams_grid["weight_decay"],
            hyperparams_grid["num_components"]):

        print(f"\nTesting: init_lr={init_lr}, weight_decay={weight_decay}, num_components={num_components}")

        # Crea una nuova istanza del modello per ogni combinazione(c'è un parametro inizializzato qui,devo fare così)
        model = RNADE(input_dim, hidden_units, num_components)
        trained_model = train_rnade(model, train_data, valid_data,
                                    num_epochs=1000, batch_size=100,
                                    init_lr=init_lr, weight_decay=weight_decay, patience=40)

        # Valutazione sul set di validazione
        if isinstance(train_data, torch.Tensor):
            valid_tensor = valid_data.clone().detach().float()
        else:
            valid_tensor = torch.tensor(valid_data, dtype=torch.float32)
        
        trained_model.eval()
        with torch.no_grad():
            val_ll = trained_model(valid_tensor).mean().item()
        print(f"Validation LL: {val_ll:.4f}")

        if val_ll > best_val_ll:
            best_val_ll = val_ll
            best_params = {"init_lr": init_lr, "weight_decay": weight_decay, "num_components": num_components}

    print(f"\nBest hyperparameters: {best_params} with Validation LL: {best_val_ll:.4f}")
    return best_params, best_val_ll


def load_dataset(file_name,sep=','):
    """
    Legge un file CSV presente nell'ambiente Colab utilizzando il separatore ';'
    e restituisce il DataFrame.

    Parametri:
      - file_name: stringa, nome del file (ad esempio 'winequality-red.csv')

    Restituisce:
      - dataset: DataFrame pandas con i dati letti
    """
    import pandas as pd
    dataset = pd.read_csv(file_name, sep=sep)
    print(f"File '{file_name}' caricato correttamente. Dimensione: {dataset.shape}")
    print("Prime 5 righe:")
    print(dataset.head())
    return dataset


def preprocess_data(df, columns_to_remove):
    """
    Rimuove colonne discrete specificate e una variabile da ogni coppia con correlazione > 0.98.

    Parametri:
      - df: DataFrame pandas con i dati
      - columns_to_remove: lista delle colonne discrete da eliminare

    Restituisce:
      - DataFrame processato con le colonne eliminate
    """
    # Rimuovi colonne discrete
    df = df.drop(columns=columns_to_remove, errors='ignore')

    # Calcola la matrice di correlazione di Pearson
    corr_matrix = df.corr().abs()

    # Trova le coppie di colonne con correlazione > 0.98
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.98)]

    # Rimuovi una colonna per ogni coppia correlata
    df = df.drop(columns=to_drop, errors='ignore')

    print(f"Colonne rimosse per alta correlazione: {to_drop}")
    return df


from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.1,random_state=123):
    """
    Divide i dati in un set di addestramento e un set di validazione/test

    Parametri:
      - data: array NumPy [N, D] dei dati.
      - test_size: proporzione del set di validazione (default 0.1 per il 10%).
      - random_state: seed per la riproducibilità (default 42).

    Restituisce:
      - train_data: array NumPy [N_train, D] dei dati di addestramento.
      - test_data: array NumPy [N_valid, D] dei dati di validazione.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"Dimensione del set di addestramento: {train_data.shape}")
    print(f"Dimensione del set di validazione: {test_data.shape}")
    return train_data, test_data


def normalize_data(train_data, test_data):
    """
    Normalizza i dati sottraendo la media e dividendo per la deviazione standard.

    Parametri:
      - train_data: array NumPy [N_train, D] dei dati di addestramento.
      - test_data: array NumPy [N_test, D] dei dati di test.

    Restituisce:
      - train_data_normalized: array NumPy normalizzato dei dati di addestramento.
      - test_data_normalized: array NumPy normalizzato dei dati di test.
    """
    # Calcola la media e la deviazione standard per il set di addestramento
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    # Normalizza i dati di addestramento
    train_data_normalized = (train_data - mean) / std

    # Normalizza i dati di test utilizzando la media e la deviazione standard del set di addestramento
    test_data_normalized = (test_data - mean) / std

    return train_data_normalized, test_data_normalized


def train_rnade_finale(model, train_data, best_cv_val_ll, num_epochs=500, batch_size=100, #cambia l'early stopping, vedi 7.3.1
                init_lr=0.025, weight_decay=0.001):
    """
    Addestra il modello RNADE utilizzando SGD sulla negativa log-likelihood.
    Viene applicato il weight decay solo al parametro condiviso W e,
    come suggerito in Uria et al. (2013), i gradienti dei parametri della media vengono scalati per σ.

    L'early stopping si basa sul fatto che se la training loss diventa maggiore della migliore loss ottenuta in
    cross validation (best_cv_val_loss), l'addestramento viene interrotto.

    Parametri:
      - train_data: array NumPy [N_train, D] dei dati di training.
      - best_cv_val_loss: migliore loss ottenuta in cross validation.
      - num_epochs: numero massimo di epoche (default 500).
      - batch_size: dimensione del minibatch (default 100).
      - init_lr: learning rate iniziale.
      - weight_decay: regolarizzazione per il parametro W.

    Restituisce il modello con lo stato del training migliore (in termini di loss sul training set).
    """
    best_cv_val_loss= -best_cv_val_ll
    if isinstance(train_data, torch.Tensor):
        train_tensor = train_data.clone().detach().float()
    else:
        train_tensor = torch.tensor(train_data, dtype=torch.float32)
    #train_tensor = torch.tensor(train_data, dtype=torch.float32)


    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Il dataloader distribuisce i dati in mini-batch

    # Applica weight decay solo al parametro condiviso W
    optimizer = optim.SGD([
        {'params': [model.W], 'weight_decay': weight_decay},
        {'params': [p for name, p in model.named_parameters() if name != 'W'], 'weight_decay': 0.0}
    ], lr=init_lr)

    best_train_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 10  # utilizziamo 10 minibatch per epoca, come specificato
        for _ in range(num_batches):
            indices = np.random.choice(len(train_dataset), batch_size, replace=True)
            batch = train_tensor[indices]
            order = list(range(model.input_dim))  # ordine naturale

            log_prob = model(batch)
            loss = -torch.mean(log_prob)

            optimizer.zero_grad()
            loss.backward()

            # Scaling del gradiente per i parametri della media:
            for d in range(model.input_dim):
                sigma_d = model.last_sigma[d].detach().mean(dim=0)  # shape: [num_components]
                if model.b_mu[d].grad is not None:
                    model.b_mu[d].grad.mul_(sigma_d)
                if model.V_mu[d].grad is not None:
                    model.V_mu[d].grad.mul_(sigma_d.unsqueeze(0))

            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= (num_batches * batch_size) #così ti riporti alla media
        # Aggiorna il learning rate con decrescita lineare
        current_lr = init_lr * (1 - (epoch + 1) / num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, LR: {current_lr:.5f}")

        # Aggiornamento del modello migliore in base alla training loss
        if epoch_loss > best_train_loss:
            best_train_loss = epoch_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        # Early stopping: se la training loss supera la migliore loss ottenuta in cross validation, fermiamo l'addestramento.
        if epoch_loss < best_cv_val_loss:
            print(f"Early stopping alla epoca {epoch+1}: training loss {epoch_loss:.4f} ha superato il best CV loss {best_cv_val_loss:.4f}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model