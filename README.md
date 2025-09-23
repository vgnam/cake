# 🍰 CAKE: Context-Aware Kernel Evolution

<div align="center">

<img src="assets/cake.png" alt="CAKE" width="400">

**Adaptive Kernel Design for Bayesian Optimization using Large Language Models**

<p align="center">
  <a href="https://github.com/richardcsuwandi/cake/stargazers"><img src="https://img.shields.io/github/stars/richardcsuwandi/cake?style=social" alt="GitHub stars"></a>
  <a href="https://neurips.cc"><img src="https://img.shields.io/badge/Paper-NeurIPS%202025-blue" alt="NeurIPS 2025"></a>
</p>

[**Overview**](#overview) • [**Quick Start**](#quick-start) • [**Experiments**](#experiments)

</div>

This repository contains the official implementation of **CAKE** (Context-Aware Kernel Evolution), a novel framework that leverages large language models to adaptively evolve Gaussian Process kernel functions for Bayesian Optimization. CAKE combines evolutionary algorithms with LLM reasoning to automatically discover kernel structures that capture patterns in optimization landscapes.

## Overview

### Why CAKE

Gaussian Process (GP) kernels are critical for the performance of Bayesian Optimization, but selecting appropriate kernels requires domain expertise and is often done manually. CAKE addresses this challenge by using LLMs to guide the evolution of kernel expressions through crossover and mutation operations, enabling automatic discovery of problem-specific kernel structures without human intervention.

### How CAKE Works

CAKE operates through an evolutionary process where:

1. **Population Initialization**: Starts with a population of base kernels (SE, Matérn, Periodic, Linear, RQ)
2. **Fitness Evaluation**: Assesses kernels using Bayesian Information Criterion (BIC) on observed data
3. **LLM-Guided Operations**: Uses language models to perform crossover and mutation operations
4. **Selection**: Maintains population diversity while preserving high-performing kernels
5. **Optimization**: Uses evolved kernels for Bayesian Optimization acquisition functions


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/richardcsuwandi/cake.git
cd cake

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"

# Install dependencies (using uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r requirements.txt --python 3.9
```

### Basic Usage

```python
import torch
from cake import CAKE
from benchmark import get_objective

# Initialize CAKE
cake = CAKE(
    num_population=6,
    mutation_prob=0.7,
    model_name="gpt-4o-mini"
)

# Set up optimization problem
objective, bounds, _ = get_objective("ackley2")
train_x = torch.rand(10, 2) * 2 - 1
train_y = objective(train_x)

# Run kernel evolution
best_kernel = cake.run(train_x, train_y)
print(f"Best evolved kernel: {best_kernel}")

# Use for optimization
next_x = cake.get_next_query(bounds)
```

## Experiments
To run the experiments in our paper, you can execute the following Python scripts:

```bash
# Run on synthetic optimization functions
python exp.py 

# Run on hyperparameter optimization tasks
python hpobench_exp.py
```

## Requirements

- Python 3.9+
- OpenAI API key (or compatible LLM API)
- Dependencies: PyTorch, BoTorch, GPyTorch, OpenAI

## LLM Configuration

CAKE supports OpenAI GPT models and compatible APIs. Configure your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

For other LLM providers, modify the CAKE initialization:
```python
cake = CAKE(
    model_name="your-model",
    api_base="your-api-endpoint"  # if using non-OpenAI API
)
```

## Citation

If you use CAKE in your research, please cite our paper:

*Coming soon!*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
