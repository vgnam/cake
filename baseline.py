import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

from svm import compute_kernel_matrix, compute_cka, fit_svm_model, BASE_KERNEL_FUNCTIONS


def evaluate_fixed_kernels(X_train, y_train, X_test, y_test):
    """
    Evaluate each base kernel individually using CKA and SVM test accuracy.
    Returns:
        dict: {kernel_name: {"cka": float, "accuracy": float}}
    """
    results = {}
    for kernel_name in BASE_KERNEL_FUNCTIONS:
        try:
            model, cka, acc = fit_svm_model(X_train, y_train, kernel_name, X_test, y_test)
            results[kernel_name] = {"cka": cka, "accuracy": acc}
        except Exception as e:
            results[kernel_name] = {"cka": 0.0, "accuracy": 0.0, "error": str(e)}
    return results


def evaluate_grid_search(X_train, y_train, X_test, y_test):
    """
    Run GridSearchCV over kernel type and hyperparameters.
    Returns:
        dict: Best parameters, CV score, and test accuracy.
    """
    param_grid = [
        {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01, 0.1]},
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
        {"kernel": ["poly"], "C": [0.1, 1, 10], "degree": [2, 3, 4]},
        {"kernel": ["sigmoid"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    ]

    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    test_acc = grid.score(X_test, y_test)

    return {
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_accuracy": test_acc
    }


def evaluate_random_kernel(X_train, y_train, X_test, y_test, n_trials=10):
    """
    Randomly select kernel expressions and evaluate CKA and accuracy.
    Returns:
        dict: Best random kernel, CKA, and accuracy.
    """
    kernels = list(BASE_KERNEL_FUNCTIONS.keys())
    operators = ["+", "*"]

    best_cka = -1
    best_result = None

    for _ in range(n_trials):
        # random single or composite kernel
        if np.random.rand() < 0.5:
            kernel_expr = np.random.choice(kernels)
        else:
            k1, k2 = np.random.choice(kernels, size=2, replace=True)
            op = np.random.choice(operators)
            kernel_expr = f"{k1} {op} {k2}"

        try:
            model, cka, acc = fit_svm_model(X_train, y_train, kernel_expr, X_test, y_test)
            if cka > best_cka:
                best_cka = cka
                best_result = {"kernel": kernel_expr, "cka": cka, "accuracy": acc}
        except Exception:
            continue

    return best_result


if __name__ == "__main__":
    from benchmark import get_dataset

    dataset = "iris"
    X_train, X_test, y_train, y_test = get_dataset(dataset)

    print(f"Dataset: {dataset}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("=" * 50)

    # fixed kernels
    print("\n--- Fixed Kernel Evaluation ---")
    fixed = evaluate_fixed_kernels(X_train, y_train, X_test, y_test)
    for k, v in fixed.items():
        print(f"  {k}: CKA = {v['cka']:.4f}, Accuracy = {v['accuracy']:.4f}")

    # grid search
    print("\n--- Grid Search ---")
    grid = evaluate_grid_search(X_train, y_train, X_test, y_test)
    print(f"  Best params: {grid['best_params']}")
    print(f"  CV score: {grid['best_cv_score']:.4f}")
    print(f"  Test accuracy: {grid['test_accuracy']:.4f}")

    # random kernel
    print("\n--- Random Kernel ---")
    rand = evaluate_random_kernel(X_train, y_train, X_test, y_test)
    if rand:
        print(f"  Best: {rand['kernel']}, CKA = {rand['cka']:.4f}, Accuracy = {rand['accuracy']:.4f}")