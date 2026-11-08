from sklearn.svm import SVC


def train_svm(X, y, C=1.0):
    """Entraîne un SVM linéaire et renvoie le modèle."""
    model = SVC(kernel="linear", C=C)
    model.fit(X, y)
    return model
