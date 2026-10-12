# Rilevare l'entanglement di due qubit con una rete neurale

Macroargomento: machine learning e reti neurali.

**Problema.** Dati i 15 valori attesi delle misure di Pauli su due qubit, dire se lo stato è entangled (1) o separabile (0).
L'etichetta è esatta (criterio di Peres-Horodecki). Un classificatore lineare sulle feature di Pauli è un "testimone di
entanglement" e sui dati generici non supera il 50 %; un MLP arriva al 95 %.

## File

| file | contenuto |
|---|---|
| `qubit.py` | le funzioni di fisica: Pauli, feature, trasposta parziale, criterio di Peres-Horodecki, generatori di stati |
| `progetto.ipynb` | il lavoro completo in tre parti: dataset, modelli lineari (le classi non sono linearmente separabili), MLP |
| `presentazione.pptx` | le slide per l'esame |

`dataset.csv` (38 248 stati, 15 feature + etichetta + famiglia) è prodotto dalla parte 1 del notebook e incluso per
comodità: eseguendo il notebook viene rigenerato identico (seed fissati).

## Esecuzione

```
pip install -r requirements.txt
jupyter notebook
```

Tempo di esecuzione su CPU: circa 3 minuti (la maggior parte nel confronto fra architetture e regolarizzazioni della parte 3).

## Riferimenti

- A. Peres, PRL 77, 1413 (1996); M. Horodecki, P. Horodecki, R. Horodecki, Phys. Lett. A 223, 1 (1996) — il criterio.
- S. Lu et al., "Separability-entanglement classifier via machine learning", PRA 98, 012315 (2018) — stesso problema con
  reti neurali, dataset generato.
