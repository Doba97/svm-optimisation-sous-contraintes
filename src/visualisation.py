import matplotlib.pyplot as plt


def plot_data(X, y):
    """Visualise un jeu de données à deux variables."""
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.xlabel("Variable 1")
    plt.ylabel("Variable 2")
    plt.title("Données de classification")
    plt.show()
