from sklearn.datasets import make_classification
from src.svm_experiment import train_svm


def test_train_svm():
    X, y = make_classification(
        n_samples=40,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42
    )
    model = train_svm(X, y)
    assert model.score(X, y) >= 0
