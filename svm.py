import numpy as np
import re
from sklearn.svm import SVC
from sklearn.metrics.pairwise import (
    rbf_kernel,
    linear_kernel,
    polynomial_kernel,
    sigmoid_kernel,
    cosine_similarity,
    laplacian_kernel,
)
from sklearn.preprocessing import label_binarize


# ── Base kernel functions ────────────────────────────────────────────────────
BASE_KERNEL_FUNCTIONS = {
    "RBF": lambda X, Y=None: rbf_kernel(X, Y),
    "LINEAR": lambda X, Y=None: linear_kernel(X, Y),
    "POLY2": lambda X, Y=None: polynomial_kernel(X, Y, degree=2),
    "POLY3": lambda X, Y=None: polynomial_kernel(X, Y, degree=3),
    "POLY4": lambda X, Y=None: polynomial_kernel(X, Y, degree=4),
    "SIGMOID": lambda X, Y=None: sigmoid_kernel(X, Y),
    "COSINE": lambda X, Y=None: cosine_similarity(X, Y),
    "LAPLACIAN": lambda X, Y=None: laplacian_kernel(X, Y),
}

# ── Supported operators ──────────────────────────────────────────────────────
# +   : kernel sum          K1 + K2         (PSD if K1, K2 are PSD)
# *   : kernel product      K1 * K2         (PSD — Schur/Hadamard product)
# **n : kernel power        K ** n          (PSD if n is positive integer)
# @   : kernel composition  K1 @ K2         (matrix product, NOT always PSD — validated)
SUPPORTED_OPERATORS = ["+", "*", "**", "@"]


# ── PSD validation ───────────────────────────────────────────────────────────
def is_psd(K, tol=1e-6):
    """
    Check if a kernel matrix is positive semi-definite.
    A matrix is PSD if all eigenvalues >= -tol.

    Args:
        K (np.ndarray): Square matrix of shape (n, n).
        tol (float): Tolerance for negative eigenvalues.
    Returns:
        bool: True if the matrix is PSD.
    """
    # symmetrize to handle numerical noise
    K_sym = (K + K.T) / 2
    eigenvalues = np.linalg.eigvalsh(K_sym)
    return bool(np.all(eigenvalues >= -tol))


def make_psd(K):
    """
    Project a matrix to the nearest PSD matrix by clipping negative eigenvalues.

    Args:
        K (np.ndarray): Square matrix.
    Returns:
        K_psd (np.ndarray): Nearest PSD matrix.
    """
    K_sym = (K + K.T) / 2
    eigvals, eigvecs = np.linalg.eigh(K_sym)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ── Kernel expression parser (free-form) ─────────────────────────────────────
def compute_kernel_matrix(X, expression, Y=None):
    """
    Parse a kernel expression string and compute the resulting kernel matrix.
    Supports free-form expressions with base kernels and operators.

    Base kernels: RBF, LINEAR, POLY2, POLY3, POLY4, SIGMOID, COSINE, LAPLACIAN
    Operators: + (sum), * (Hadamard product), ** (power), @ (matrix product)
    Parentheses for grouping.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        expression (str): Kernel expression, e.g. "RBF + LINEAR", "(POLY3 * RBF) + COSINE".
        Y (np.ndarray, optional): Second input data; if None, uses X.
    Returns:
        K (np.ndarray): Kernel matrix.
    Raises:
        ValueError: If the resulting kernel matrix is not PSD.
    """
    cache = {}
    base_kernels = dict(BASE_KERNEL_FUNCTIONS)

    def _apply_op(left, op, right):
        if op == '+':
            return left + right
        elif op == '*':
            return left * right
        elif op == '@':
            return left @ right
        else:
            raise ValueError(f"Unknown operator: {op}")

    def _compute_subexpr(subexpr):
        # handle power operator: KERNEL**N
        power_match = re.match(r'^(\w+)\*\*(\d+)$', subexpr.strip())
        if power_match:
            kernel_name, power = power_match.group(1), int(power_match.group(2))
            K = base_kernels[kernel_name](X, Y)
            result = K.copy()
            for _ in range(power - 1):
                result = result * K  # Hadamard power (element-wise)
            return result

        tokens = re.findall(r'[\w]+', subexpr)
        operators = re.findall(r'[+*@]', subexpr)

        result = base_kernels[tokens[0]](X, Y)
        for i, op in enumerate(operators):
            right = base_kernels[tokens[i + 1]](X, Y)
            result = _apply_op(result, op, right)
        return result

    # handle parenthesized sub-expressions first
    pattern = r'\(([^()]+)\)'
    counter = 0
    while '(' in expression:
        for subexpr in re.findall(pattern, expression):
            if subexpr not in cache:
                sub_K = _compute_subexpr(subexpr)
                name = f'SubKernel{counter}'
                cache[subexpr] = name
                base_kernels[name] = lambda X_, Y_=None, _K=sub_K: _K
                counter += 1
            expression = expression.replace(f'({subexpr})', cache[subexpr], 1)

    K = _compute_subexpr(expression)

    # PSD validation: only for square matrices (K(X,X)), not rectangular K(X,Y)
    if K.shape[0] == K.shape[1] and not is_psd(K):
        K = make_psd(K)

    return K


# ── Centered Kernel Alignment (CKA) ─────────────────────────────────────────
def _center_kernel(K):
    """Center a kernel matrix using H = I - 11^T/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(K, L):
    """Compute the Hilbert-Schmidt Independence Criterion (HSIC)."""
    n = K.shape[0]
    Kc = _center_kernel(K)
    Lc = _center_kernel(L)
    return np.trace(Kc @ Lc) / ((n - 1) ** 2)


def compute_cka(K, y):
    """
    Compute CKA between a kernel matrix K and the label kernel L = yy^T.

    For multi-class labels, y is one-hot encoded before computing L.
    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    Args:
        K (np.ndarray): Kernel matrix of shape (n, n).
        y (np.ndarray): Labels of shape (n,).
    Returns:
        float: CKA score in [0, 1].
    """
    classes = np.unique(y)
    if len(classes) <= 2:
        y_vec = y.reshape(-1, 1).astype(float)
    else:
        y_vec = label_binarize(y, classes=classes).astype(float)

    L = y_vec @ y_vec.T  # label kernel

    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


# ── SVM fitting (for final evaluation) ──────────────────────────────────────
def fit_svm_model(X_train, y_train, kernel_expr, X_test=None, y_test=None, C=1.0):
    """
    Fit an SVM using a precomputed kernel matrix from the parsed expression.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        kernel_expr (str): Kernel expression string.
        X_test (np.ndarray, optional): Test features.
        y_test (np.ndarray, optional): Test labels.
        C (float): SVM regularization parameter.
    Returns:
        model (SVC): Fitted SVM model.
        cka (float): CKA score on training data.
        accuracy (float or None): Test accuracy if test data provided.
    """
    K_train = compute_kernel_matrix(X_train, kernel_expr)
    cka = compute_cka(K_train, y_train)

    model = SVC(kernel='precomputed', C=C)
    model.fit(K_train, y_train)

    accuracy = None
    if X_test is not None and y_test is not None:
        K_test = compute_kernel_matrix(X_test, kernel_expr, Y=X_train)
        accuracy = model.score(K_test, y_test)

    return model, cka, accuracy
