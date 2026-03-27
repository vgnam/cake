import numpy as np
from cake import CAKE
from svm import fit_svm_model
from benchmark import get_dataset
import os

os.environ["NVIDIA_NIM_API_KEY"] = "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"

import warnings
warnings.filterwarnings("ignore")


# ── Configuration ────────────────────────────────────────────────────────────
dataset_name = "digits"       # dataset to use (see benchmark.py)
num_generations = 10         # number of evolutionary generations
num_population = 8          # population size
model_name = "nvidia_nim/openai/gpt-oss-120b"  # LLM to use
REPEAT = 3                  # number of repetitions

print(f"Dataset: {dataset_name}")
print(f"Generations: {num_generations}")
print(f"LLM: {model_name}")
print("=" * 50)

for r in range(REPEAT):
    print(f"\n--- Trial {r + 1}/{REPEAT} ---")

    # load dataset
    X_train, X_test, y_train, y_test = get_dataset(dataset_name, seed=r)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # initialize CAKE
    cake = CAKE(
        num_population=num_population,
        model_name=model_name
    )
    print(f"Base kernels: {cake.base_kernels}")
    print(f"Operators: {cake.operators}")

    # run evolutionary generations
    fitness_history = []
    for gen in range(num_generations):
        best_kernel, best_cka = cake.run(X_train, y_train)
        fitness_history.append(best_cka)
        print(f"  Gen {gen + 1}: best kernel = {best_kernel}, CKA = {best_cka:.4f}")

    # final evaluation: fit SVM with the best kernel
    best_kernel, best_cka = cake.get_best_kernel()
    model, cka, test_acc = fit_svm_model(X_train, y_train, best_kernel, X_test, y_test)

    print(f"\n  Best kernel: {best_kernel}")
    print(f"  CKA score:   {cka:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    # print population
    print(f"  Population:")
    for k, v in cake.population.items():
        print(f"    {k}: CKA = {v['fitness']:.4f}")

print("\n" + "=" * 50)
print("Done.")