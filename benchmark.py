import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "iris": load_iris,
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}


def get_dataset(name, test_size=0.2, seed=42):
    """
    Load a classification dataset and return train/test splits with scaling.
    Args:
        name (str): Dataset name (iris, breast_cancer, wine, digits).
        test_size (float): Fraction of data for testing.
        seed (int): Random seed.
    Returns:
        X_train, X_test, y_train, y_test (np.ndarray): Scaled train/test splits.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")

    data = DATASETS[name]()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
