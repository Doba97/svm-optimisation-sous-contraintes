"""Génération des jeux de données synthétiques du projet.

Deux jeux de données à deux variables :
- un cas linéairement séparable (marge dure) ;
- un cas non séparable, avec du bruit dans les étiquettes (marge souple).
"""

import numpy as np
import pandas as pd


def generer_donnees_separables(n=80, graine_x1=123, graine_x2=567):
    """Cas séparable : classe +1 si X1 > X2, classe -1 sinon.

    Les deux variables sont tirées uniformément sur [0, 1], puis centrées
    et réduites avec l'écart-type empirique (ddof=1).
    """
    rng1 = np.random.RandomState(graine_x1)
    X1_brut = rng1.uniform(0, 1, n)

    rng2 = np.random.RandomState(graine_x2)
    X2_brut = rng2.uniform(0, 1, n)

    # centrage et réduction avec l'écart-type empirique (ddof=1)
    X1 = (X1_brut - X1_brut.mean()) / X1_brut.std(ddof=1)
    X2 = (X2_brut - X2_brut.mean()) / X2_brut.std(ddof=1)

    # Classe -1 si X1 <= X2, +1 sinon
    Y = np.where(X1 > X2, 1, -1)

    return pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})


def generer_donnees_non_separables(n=1000, graine=1234):
    """Cas non séparable : étiquettes bruitées autour de la diagonale V1 = V2.

    La probabilité que Y = 1 vaut 0,95 si V1 <= V2 et 0,05 sinon.
    Y prend ici les valeurs 0 et 1.
    """
    rng = np.random.RandomState(graine)

    V1 = rng.uniform(0, 1, n)
    V2 = rng.uniform(0, 1, n)

    # Probabilité de Y=1 selon la position par rapport à la diagonale
    p = np.where(V1 <= V2, 0.95, 0.05)

    Y_bruit = rng.binomial(1, p)

    return pd.DataFrame({"V1": V1, "V2": V2, "Y": Y_bruit})
