"""Le funzioni di fisica del progetto. Tutto il resto e' machine learning (Vol. 3).

- Uno stato di due qubit e' descritto da una matrice 4x4, qui chiamata rho.
- Le feature di uno stato sono le 15 medie delle misure lungo gli assi X, Y, Z:
  3 sul solo qubit A, 3 sul solo qubit B, 9 correlazioni fra A e B.
- L'etichetta (entangled / separabile) viene dal criterio di Peres-Horodecki.
"""
import numpy as np

# Matrici di Pauli: rappresentano le misure lungo i tre assi X, Y, Z di un qubit
# (esito +1 o -1). I e' "nessuna misura".
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}

# Le 15 coppie (a, b) misurabili: "XI" = X sul qubit A e nulla su B, ecc.
# "II" non c'e' perche' <I(x)I> = Tr(rho) = 1 per ogni stato: non distingue nulla.
NOMI_FEATURE = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]
OSSERVABILI = [np.kron(PAULI[n[0]], PAULI[n[1]]) for n in NOMI_FEATURE]


def feature(rho):
    """Le 15 medie delle misure: <a su A, b su B> = Tr(rho * O_ab).
    Sono numeri reali in [-1, 1] (la parte immaginaria e' zero a meno di
    arrotondamenti, quindi si tiene solo la parte reale)."""
    return np.array([np.trace(rho @ O).real for O in OSSERVABILI])


def trasposta_parziale(rho):
    """"Specchia" la parte di B della matrice: si scambiano solo gli indici di B.
    rho[i1 j1, i2 j2] -> rho[i1 j2, i2 j1]   (i = qubit A, j = qubit B)."""
    r = rho.reshape(2, 2, 2, 2)          # indici (i1, j1, i2, j2)
    r = r.transpose(0, 3, 2, 1)          # scambia j1 con j2
    return r.reshape(4, 4)


def entangled(rho):
    """Criterio di Peres-Horodecki: lo stato e' entangled se e solo se la matrice
    specchiata ha un autovalore negativo (esatto per due qubit).
    Restituisce 1 (entangled) oppure 0 (separabile)."""
    autovalori = np.linalg.eigvalsh(trasposta_parziale(rho))
    return int(autovalori.min() < -1e-12)


# --------------------------------------------------------------- generatori
def stato_misto_casuale(rng, dim=4):
    """Stato generico casuale: G G^dagger normalizzato, con G a elementi gaussiani
    complessi. Questa costruzione garantisce che rho sia uno stato valido."""
    G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


def stato_puro_casuale(rng):
    """Stato puro casuale (senza rumore): vettore casuale normalizzato, rho = |psi><psi|."""
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def stato_puro_con_rumore(rng):
    """Stato puro casuale mescolato con una quantita' casuale di rumore:
    rho = p * puro + (1-p) * rumore. Per p = 1 e' puro, per p = 0 e' solo rumore
    (separabile). Al variare di p si attraversa il confine fra le due classi."""
    p = rng.uniform(0, 1)
    return p * stato_puro_casuale(rng) + (1 - p) * np.eye(4) / 4


def stato_separabile_casuale(rng, k=3):
    """Miscela di k stati in cui A e B sono indipendenti (rho_A (x) rho_B), con
    pesi casuali che sommano a 1. E' la definizione di stato separabile:
    etichetta 0 per costruzione (il criterio lo conferma)."""
    pesi = rng.dirichlet(np.ones(k))
    rho = np.zeros((4, 4), dtype=complex)
    for w in pesi:
        rho += w * np.kron(stato_misto_casuale(rng, 2), stato_misto_casuale(rng, 2))
    return rho
