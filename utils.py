import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")


def plot_fitness_history(fitness_history, title="CKA Fitness Over Generations"):
    """
    Plot CKA fitness scores across evolutionary generations.
    Args:
        fitness_history (list[float]): CKA score per generation.
        title (str): Plot title.
    """
    plt.figure(figsize=(8, 5))
    generations = np.arange(1, len(fitness_history) + 1)
    plt.plot(generations, fitness_history, marker="o", linewidth=2, color="#4A90D9")
    plt.fill_between(generations, fitness_history, alpha=0.15, color="#4A90D9")
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("CKA Fitness", fontsize=14)
    plt.title(title, fontsize=16)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_kernel_matrix(K, title="Kernel Matrix"):
    """
    Visualize a kernel matrix as a heatmap.
    Args:
        K (np.ndarray): Kernel matrix of shape (n, n).
        title (str): Plot title.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(K, cmap="viridis", square=True, xticklabels=False, yticklabels=False)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_comparison(results, metric="cka", title="Kernel Comparison"):
    """
    Bar plot comparing kernels by a metric (CKA or accuracy).
    Args:
        results (dict): {kernel_name: {"cka": float, "accuracy": float}}.
        metric (str): "cka" or "accuracy".
        title (str): Plot title.
    """
    kernels = list(results.keys())
    values = [results[k][metric] for k in kernels]

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("husl", len(kernels))
    bars = plt.bar(kernels, values, color=colors, edgecolor="black", linewidth=0.5)
    plt.ylabel(metric.upper(), fontsize=14)
    plt.title(title, fontsize=16)
    plt.ylim(0, 1.05)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.show()