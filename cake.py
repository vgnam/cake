import numpy as np
import time
from litellm import completion

from svm import compute_kernel_matrix, compute_cka, fit_svm_model, is_psd

# prompts for the LLM — free-form kernel design
SYSTEM_PROMPT_TEMPLATE = """
You are an expert in machine learning, specializing in Support Vector Machines and kernel methods. Here is a summary of the classification dataset:
{dataset_summary}

Your task is to design a kernel function for an SVM classifier that best separates the classes.

Available base kernels: {base_kernels}
Available operators to combine kernels:
  +   : kernel sum (K1 + K2)
  *   : element-wise kernel product / Hadamard product (K1 * K2)
  **n : element-wise power (K**2, K**3, etc.)
  @   : matrix product (K1 @ K2)
  ()  : parentheses for grouping

You are free to compose kernels creatively using any combination of the above.
The kernel will be evaluated using Centered Kernel Alignment (CKA), scored in [0, 1].
The resulting kernel matrix must be positive semi-definite (PSD) — invalid kernels will be rejected.
"""

CROSSOVER_PROMPT_TEMPLATE = """
You are given two parent kernels and their CKA fitness scores:
  Parent 1: {parent_kernel1} (CKA = {fitness1:.4f})
  Parent 2: {parent_kernel2} (CKA = {fitness2:.4f})

Design a new kernel by creatively combining, merging, or restructuring the parent kernels.
You may use any operators: +, *, **, @, and parentheses.
You may also introduce new base kernels from: {base_kernels}.

Think step by step about what patterns each parent captures and how to combine their strengths.

Please respond EXACTLY in this format:
Kernel: <your kernel expression using only the base kernel names and operators>
Analysis: <brief explanation>
"""

MUTATION_PROMPT_TEMPLATE = """
You are given a kernel and its CKA fitness score:
  {kernel} (CKA = {fitness:.4f})

Propose an improved kernel by modifying, extending, or restructuring the expression.
You may:
  - Replace any base kernel with another from: {base_kernels}
  - Add, remove, or change operators (+, *, **, @)
  - Restructure with parentheses
  - Make the expression more complex or simpler

Think about what might be limiting this kernel and how to improve it.

Please respond EXACTLY in this format:
Kernel: <your kernel expression using only the base kernel names and operators>
Analysis: <brief explanation>
"""

class CAKE:
    def __init__(
            self,
            num_crossover=1, # number of crossovers operation
            mutation_prob=0.7,
            num_population=4, # number of kernels to keep in the population
            model_name="nvidia_nim/openai/gpt-oss-120b", # LLM to use
            temperature=1.0, # LLM temperature
            top_p=1.0, # LLM top_p parameter
        ):
        self.num_crossover = num_crossover
        self.mutation_prob = mutation_prob
        self.num_population = num_population

        # define base kernels (expanded set)
        self.base_kernels = ["RBF", "LINEAR", "POLY2", "POLY3", "POLY4", "SIGMOID", "COSINE", "LAPLACIAN"]
        self.operators = ["+", "*", "**", "@"]

        # initial population
        self.population = {kernel: {} for kernel in self.base_kernels}

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, message, system_prompt):
        if not message:
            return "Your input is empty."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        for attempt in range(5):
            try:
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=60,
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[LLM ERROR] Attempt {attempt + 1}/5: {e}")
                if attempt == 4:
                    raise RuntimeError("Failed to get LLM response after 5 attempts") from e
                time.sleep(10)
        return ""

    @staticmethod
    def parse_response(response):
        """
        Function to parse the response from the LLM.
        Args:
            response (str): response from the LLM.
        """
        kernel_start = response.find("Kernel: ") + len("Kernel: ")
        kernel_end = response.find("\n", kernel_start)
        kernel = response[kernel_start:kernel_end].strip()

        analysis_start = response.find("Analysis: ") + len("Analysis: ")
        analysis = response[analysis_start:]
        return kernel, analysis

    def update_data(self, X, y):
        """
        Function to update the training data and system prompt.
        Args:
            X (np.ndarray): training features of shape (n_samples, n_features).
            y (np.ndarray): training labels of shape (n_samples,).
        """
        self.X = X
        self.y = y

        # build dataset summary for the system prompt
        n_samples, n_features = X.shape
        classes, counts = np.unique(y, return_counts=True)
        class_dist = ", ".join([f"class {c}: {cnt}" for c, cnt in zip(classes, counts)])
        feature_stats = f"mean={X.mean(axis=0)[:5].round(3).tolist()}, std={X.std(axis=0)[:5].round(3).tolist()}"

        dataset_summary = (
            f"- {n_samples} samples, {n_features} features\n"
            f"- Classes: {class_dist}\n"
            f"- Feature statistics (first 5): {feature_stats}"
        )
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            dataset_summary=dataset_summary,
            base_kernels=self.base_kernels,
            operators=self.operators
        )

    def compute_fitness(self):
        """
        Function to compute the CKA fitness scores for the kernels in the population.
        """
        for kernel in list(self.population.keys()):
            try:
                K = compute_kernel_matrix(self.X, kernel)
                cka = compute_cka(K, self.y)
                self.population[kernel] = {"fitness": cka}
            except Exception:
                self.population[kernel] = {"fitness": 0.0}

        # get the fitness values and compute selection probabilities
        fitness_values = np.array([self.population[k]["fitness"] for k in self.population])
        # shift to avoid negative values for softmax
        fitness_shifted = fitness_values - fitness_values.max()
        exp_vals = np.exp(fitness_shifted)
        self.population_prob = exp_vals / exp_vals.sum()

    def generate_kernels(self):
        """
        Function to generate new kernels using crossover and mutation.
        LLM is free to propose any kernel expression; PSD is validated after computation.
        """
        # crossover step
        mating_pool = list(self.population.keys())
        for _ in range(self.num_crossover):
            parent_kernel1, parent_kernel2 = np.random.choice(
                mating_pool, size=2, p=self.population_prob, replace=False
            )
            try:
                response = self(CROSSOVER_PROMPT_TEMPLATE.format(
                    parent_kernel1=parent_kernel1,
                    parent_kernel2=parent_kernel2,
                    fitness1=self.population[parent_kernel1]["fitness"],
                    fitness2=self.population[parent_kernel2]["fitness"],
                    base_kernels=self.base_kernels
                ), self.system_prompt)
                kernel, analysis = self.parse_response(response)
            except Exception:
                # fallback: simple combination with PSD-safe operators
                safe_ops = ["+", "*"]
                kernel = f"{parent_kernel1} {np.random.choice(safe_ops)} {parent_kernel2}"
            try:
                if len(kernel) < 60:
                    K = compute_kernel_matrix(self.X, kernel)  # PSD validated inside
                    cka = compute_cka(K, self.y)
                    self.population[kernel] = {"fitness": cka}
                    print(f"  [CROSSOVER] {kernel} → CKA = {cka:.4f}")
            except Exception as e:
                print(f"  [CROSSOVER REJECTED] {kernel}: {e}")
                continue

        # mutation step
        if np.random.rand() < self.mutation_prob:
            # select the fittest kernel to mutate
            kernel_to_mutate = max(self.population, key=lambda x: self.population[x]["fitness"])
            try:
                response = self(MUTATION_PROMPT_TEMPLATE.format(
                    kernel=kernel_to_mutate,
                    fitness=self.population[kernel_to_mutate]["fitness"],
                    base_kernels=self.base_kernels
                ), self.system_prompt)
                kernel, analysis = self.parse_response(response)
                K = compute_kernel_matrix(self.X, kernel)  # PSD validated inside
                cka = compute_cka(K, self.y)
                self.population[kernel] = {"fitness": cka}
                print(f"  [MUTATION] {kernel_to_mutate} → {kernel}, CKA = {cka:.4f}")
            except Exception as e:
                print(f"  [MUTATION REJECTED] {kernel}: {e}")

    def update_population(self):
        """
        Function to update the population by selecting the fittest kernels.
        """
        # sort population by fitness (descending) and keep the top kernels
        sorted_pop = sorted(self.population.items(), key=lambda x: x[1]["fitness"], reverse=True)
        self.population = dict(sorted_pop[:self.num_population])

        # recompute selection probabilities
        fitness_values = np.array([self.population[k]["fitness"] for k in self.population])
        fitness_shifted = fitness_values - fitness_values.max()
        exp_vals = np.exp(fitness_shifted)
        self.population_prob = exp_vals / exp_vals.sum()

    def get_best_kernel(self):
        """
        Function to return the best kernel in the population.
        Returns:
            str: the best kernel expression.
            float: the CKA fitness score.
        """
        best = max(self.population, key=lambda x: self.population[x]["fitness"])
        return best, self.population[best]["fitness"]

    def run(self, X, y):
        """
        Function to run CAKE for SVM kernel selection.
        Args:
            X (np.ndarray): training features.
            y (np.ndarray): training labels.
        Returns:
            str: the best kernel selected by CAKE.
            float: the CKA fitness score of the best kernel.
        """
        self.update_data(X, y)
        self.compute_fitness()
        self.generate_kernels()
        self.update_population()
        return self.get_best_kernel()
