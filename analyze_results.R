results_5_dim_mnvtnorm_5000 
res_mean <- apply(results_5_dim_mnvtnorm_5000[,-c(1:2)],2,mean)
res_mean
library(tidyverse)  # include ggplot2, tidyr, dplyr

# Esempio: leggi CSV con risultati
# Sostituisci "results.csv" col percorso reale
df <- results_20_dim_exp_10000

# Controlla le colonne; ci aspettiamo almeno:
# seed, perm, gmm_rel_L1, kde_rel_L1, rnade_rel_L1, realnvp_rel_L1

# Rimuovi o ignora colonne non necessarie (es. perm) se vuoi
# Poi pivot longer su colonne rel_L1:
df_long <- df %>%
  select(seed,gmm_rel_L1_median,kde_rel_L1_median,
         rnade_rel_L1_median,realnvp_rel_L1_median) %>%
  pivot_longer(
    cols = -seed,
    names_to = "method",
    values_to = "rel_L1"
  )

# Spesso i nomi sono 'gmm_rel_L1'; vogliamo pulire method in etichette più leggibili:
df_long <- df_long %>%
  mutate(method = case_when(
    str_detect(method, "^gmm") ~ "GMM",
    str_detect(method, "^kde") ~ "KDE",
    str_detect(method, "^rnade") ~ "RNADE",
    str_detect(method, "^realnvp") ~ "RealNVP",
    TRUE ~ method
  ))


library(ggplot2)

p <- ggplot(df_long, aes(x = method, y = rel_L1, fill = method)) +
  geom_boxplot(outlier.alpha = 0.5, width = 0.6) +
  # Se vuoi mostrare i punti, puoi aggiungere jitter:
  # geom_jitter(aes(color = method), width = 0.15, alpha = 0.3, show.legend = FALSE) +
  scale_y_continuous(trans = "log10") +  # se rel_L1 varia molto, scala log; rimuovi se non serve
  scale_fill_brewer(palette = "Set2") +   # palette carina, cambia se vuoi
  labs(
    title = "Scenario B; D = 20; N = 10000",
    x = "Metodo",
    y = "Errore relativo L1",
    fill = "Metodo"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 0, vjust = 0.5),
    plot.title = element_text(hjust = 0.5)
  )

# Mostra il plot
print(p)

