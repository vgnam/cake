# 🍰 CAKE: Context-Aware Kernel Evolution for SVM

<div align="center">

**LLM-Guided Kernel Design for Support Vector Machines using Centered Kernel Alignment**

</div>

We present **CAKE** (Context-Aware Kernel Evolution), a framework that leverages large language models to adaptively evolve SVM kernel functions for classification tasks. CAKE uses an evolutionary algorithm guided by LLM reasoning to discover kernel structures that maximize **Centered Kernel Alignment (CKA)** with the label structure.

## Overview

### Method

| Step                       | Description                                                                                   |
|----------------------------|-----------------------------------------------------------------------------------------------|
| **Initialization**         | Initialize the population with base kernels: RBF, LINEAR, POLY, SIGMOID                      |
| **Fitness evaluation**     | Evaluate each kernel using CKA (alignment with the label kernel)                              |
| **LLM-guided evolution**   | Apply crossover and mutation to kernel expressions, guided by LLM reasoning                   |
| **Selection**              | Retain high-CKA kernels and proceed to the next generation                                    |

### Centered Kernel Alignment (CKA)

CKA measures how well a candidate kernel matrix **K** aligns with the ideal label kernel **yy^T**:

```
CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
```

A higher CKA score means the kernel better captures the class structure — making it a principled, training-free fitness metric for kernel selection.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/richardcsuwandi/cake.git
cd cake

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from cake import CAKE
from benchmark import get_dataset

# Load dataset
X_train, X_test, y_train, y_test = get_dataset("iris")

# Initialize CAKE
cake = CAKE(num_population=4, model_name="gpt-4o-mini")

# Run kernel evolution (multiple generations)
for gen in range(5):
    best_kernel, cka = cake.run(X_train, y_train)
    print(f"Gen {gen+1}: {best_kernel} (CKA={cka:.4f})")
```

## Experiments

```bash
# Run evolutionary kernel search
python exp.py

# Run baselines (fixed kernels, grid search, random)
python baseline.py
```

## Available Datasets

| Dataset        | Samples | Features | Classes |
|---------------|---------|----------|---------|
| `iris`         | 150     | 4        | 3       |
| `breast_cancer`| 569     | 30       | 2       |
| `wine`         | 178     | 13       | 3       |
| `digits`       | 1797    | 64       | 10      |

## Requirements

- Python 3.9+
- OpenAI API key (or compatible LLM API)
- Dependencies: NumPy, scikit-learn, matplotlib, seaborn, openai

## LLM Configuration

```bash
export OPENAI_API_KEY="your-api-key"
```

For other LLM providers:
```python
cake = CAKE(model_name="your-model", api_base="your-api-endpoint")
```

## License

This project is licensed under the [MIT License](LICENSE).
