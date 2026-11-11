"""Fonctions de visualisation : données, frontière de décision, marge, vecteurs de support."""

import numpy as np
import matplotlib.pyplot as plt

# Mêmes couleurs que les schémas du rapport : rouge = +1, vert = -1
COULEURS = {1: "red", -1: "green"}


def tracer_donnees(X, y, ax=None, titre="", xlabel="X1", ylabel="X2", taille=25):
    """Nuage de points coloré par classe (y dans {-1, +1})."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    for classe in (-1, 1):
        masque = (y == classe)
        ax.scatter(X[masque, 0], X[masque, 1], c=COULEURS[classe], s=taille,
                   label=f"Classe {classe:+d}", alpha=0.8)
    ax.set_title(titre)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def tracer_svm(modele, X, y, ax=None, titre="", xlabel="X1", ylabel="X2",
               taille=25, marge=True, montrer_sv=True):
    """Frontière de décision, bords de la marge et vecteurs de support d'un SVM linéaire.

    - trait plein : f(x) = 0 (hyperplan)
    - traits pointillés : f(x) = -1 et f(x) = +1 (bords de la marge)
    - points cerclés : vecteurs de support (désactivable avec montrer_sv=False)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    tracer_donnees(X, y, ax=ax, titre=titre, xlabel=xlabel, ylabel=ylabel, taille=taille)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
    grille = np.c_[xx.ravel(), yy.ravel()]
    Z = modele.decision_function(grille).reshape(xx.shape)

    niveaux = [-1, 0, 1] if marge else [0]
    styles = ["--", "-", "--"] if marge else ["-"]
    ax.contour(xx, yy, Z, levels=niveaux, linestyles=styles, colors="k", linewidths=1.2)

    if montrer_sv:
        sv = modele.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=140, facecolors="none", edgecolors="k",
                   linewidths=1.5, label="Vecteurs de support")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="best", fontsize=8)
    return ax
