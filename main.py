import os
import uuid
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pettingzoo.atari import boxing_v2
from typing import List, Tuple, Dict, Any

# Global constants for default setup
DEFAULT_POP_SIZE = 20
DEFAULT_GENERATIONS = 50
DEFAULT_ELITE_K = 4
DEFAULT_MUTATION_RATE = 0.9
DEFAULT_OUTPUT_UNITS = 18  # Number of actions in boxing_v2

# --------------------------
# Flexible Genome Definition
# --------------------------
class Genome:
    def __init__(self, 
                 input_size: int,
                 hidden_units: int,
                 output_units: int = DEFAULT_OUTPUT_UNITS,
                 activation: str = "tanh",
                 extra_layer: bool = False,
                 img_size: Tuple[int, int] = (20, 20)):
        self.id = str(uuid.uuid4())
        self.input_size = input_size  # = img_size[0]*img_size[1]
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.activation = activation  # "tanh", "relu", or "sigmoid"
        self.extra_layer = extra_layer
        self.img_size = img_size  # resolution to use in preprocessing
        # Initialize weights based on architecture
        self.fitness = 0.0
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Two-layer network
        if not self.extra_layer:
            self.w1 = np.random.randn(self.input_size, self.hidden_units) * 0.5
            self.b1 = np.zeros(self.hidden_units)
            self.w2 = np.random.randn(self.hidden_units, self.output_units) * 0.5
            self.b2 = np.zeros(self.output_units)
        else:
            # Three-layer network: input -> hidden1 -> hidden2 -> output
            self.w1 = np.random.randn(self.input_size, self.hidden_units) * 0.5
            self.b1 = np.zeros(self.hidden_units)
            self.w2 = np.random.randn(self.hidden_units, self.hidden_units) * 0.5
            self.b2 = np.zeros(self.hidden_units)
            self.w3 = np.random.randn(self.hidden_units, self.output_units) * 0.5
            self.b3 = np.zeros(self.output_units)
    
    def mutate(self, mutation_scale: float = 0.05):
        # Mutate each parameter: add Gaussian noise scaled by mutation_scale
        for param in self.get_params():
            param += np.random.randn(*param.shape) * mutation_scale

    def clone(self) -> 'Genome':
        child = Genome(self.input_size, self.hidden_units, self.output_units, self.activation, self.extra_layer, self.img_size)
        # Copy all weight matrices
        if not self.extra_layer:
            child.w1 = np.copy(self.w1)
            child.b1 = np.copy(self.b1)
            child.w2 = np.copy(self.w2)
            child.b2 = np.copy(self.b2)
        else:
            child.w1 = np.copy(self.w1)
            child.b1 = np.copy(self.b1)
            child.w2 = np.copy(self.w2)
            child.b2 = np.copy(self.b2)
            child.w3 = np.copy(self.w3)
            child.b3 = np.copy(self.b3)
        return child

    def get_params(self) -> List[np.ndarray]:
        if not self.extra_layer:
            return [self.w1, self.b1, self.w2, self.b2]
        else:
            return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def save_weights(self, filepath: str):
        # Save the genome parameters in a compressed npz file.
        if not self.extra_layer:
            np.savez(filepath, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        else:
            np.savez(filepath, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)

# -----------------------
# Activation Helper
# -----------------------
def apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "tanh":
        return np.tanh(x)
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError("Unknown activation: " + activation)

# -----------------------
# Preprocessing Function
# -----------------------
def preprocess(obs, img_size: Tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, img_size)
    normalized = resized / 255.0
    return normalized.flatten()

# -----------------------
# Genome Policy Function
# -----------------------
def genome_policy(observation: np.ndarray, genome: Genome) -> int:
    x = preprocess(observation, genome.img_size)
    if not genome.extra_layer:
        h = apply_activation(np.dot(x, genome.w1) + genome.b1, genome.activation)
        logits = np.dot(h, genome.w2) + genome.b2
    else:
        h1 = apply_activation(np.dot(x, genome.w1) + genome.b1, genome.activation)
        h2 = apply_activation(np.dot(h1, genome.w2) + genome.b2, genome.activation)
        logits = np.dot(h2, genome.w3) + genome.b3
    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return np.random.choice(len(probs), p=probs)

# -----------------------
# Simulation and Evaluation
# -----------------------
def simulate_match(genome_a: Genome, genome_b: Genome) -> Tuple[float, float]:
    # Set up the environment; change render_mode to "human" for visualization if needed.
    env = boxing_v2.parallel_env(render_mode=None)
    obs, _ = env.reset()
    agents = env.agents
    total_reward_a, total_reward_b = 0, 0

    while env.agents:
        actions = {
            agents[0]: genome_policy(obs[agents[0]], genome_a),
            agents[1]: genome_policy(obs[agents[1]], genome_b),
        }
        obs, rewards, terms, truncs, _ = env.step(actions)
        total_reward_a += rewards.get(agents[0], 0)
        total_reward_b += rewards.get(agents[1], 0)
        if all(terms.values()) or all(truncs.values()):
            break
    env.close()
    return total_reward_a, total_reward_b

def evaluate_population(pop_a: List[Genome], pop_b: List[Genome]) -> Tuple[float, float]:
    # Reset fitness values before evaluation.
    for genome in pop_a + pop_b:
        genome.fitness = 0
    total_rewards_a, total_rewards_b = 0, 0

    # Evaluate in a pairwise match.
    for genome_a, genome_b in zip(pop_a, pop_b):
        reward_a, reward_b = simulate_match(genome_a, genome_b)
        genome_a.fitness += reward_a
        genome_b.fitness += reward_b
        total_rewards_a += reward_a
        total_rewards_b += reward_b

    avg_reward_a = total_rewards_a / len(pop_a)
    avg_reward_b = total_rewards_b / len(pop_b)
    return avg_reward_a, avg_reward_b

def select_and_reproduce(pop: List[Genome], freeze: bool, elite_k: int, mutation_rate: float) -> List[Genome]:
    pop.sort(key=lambda g: g.fitness, reverse=True)
    elites = pop[:elite_k]
    new_pop = []
    if freeze:
        # If frozen, simply clone the current population without mutation.
        new_pop = [g.clone() for g in pop]
    else:
        while len(new_pop) < len(pop):
            parent = random.choice(elites)
            child = parent.clone()
            if random.random() < mutation_rate:
                child.mutate()
            new_pop.append(child)
    return new_pop

# -----------------------
# Evolution Session Function
# -----------------------
def evolution_session(exp_config: Dict[str, Any]):
    """
    Runs an evolutionary session based on the provided configuration.
    exp_config keys:
      - name: Experiment name.
      - pop_size: Population size (default DEFAULT_POP_SIZE).
      - generations: Number of generations (default DEFAULT_GENERATIONS).
      - elite_k: Number of elites (default DEFAULT_ELITE_K).
      - mutation_rate: Mutation rate (default DEFAULT_MUTATION_RATE).
      - freeze: Which population to freeze ("a", "b", or None).
      - genome_params_a: Dict of parameters for Population A's Genome.
      - genome_params_b: Dict of parameters for Population B's Genome.
    """
    name = exp_config.get("name", "experiment")
    pop_size = exp_config.get("pop_size", DEFAULT_POP_SIZE)
    generations = exp_config.get("generations", DEFAULT_GENERATIONS)
    elite_k = exp_config.get("elite_k", DEFAULT_ELITE_K)
    mutation_rate = exp_config.get("mutation_rate", DEFAULT_MUTATION_RATE)
    freeze = exp_config.get("freeze", None)  # "a", "b", or None
    
    # Create a result directory for this experiment.
    results_dir = os.path.join("results", name)
    os.makedirs(results_dir, exist_ok=True)
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Determine image size and input dimension from genome parameters.
    # For both populations, if not provided, use defaults.
    genome_params_a = exp_config.get("genome_params_a", {})
    genome_params_b = exp_config.get("genome_params_b", genome_params_a.copy())
    
    # Set defaults if not provided
    default_img_size = (20, 20)
    genome_params_a.setdefault("img_size", default_img_size)
    genome_params_b.setdefault("img_size", default_img_size)
    input_size_a = genome_params_a["img_size"][0] * genome_params_a["img_size"][1]
    input_size_b = genome_params_b["img_size"][0] * genome_params_b["img_size"][1]
    # Also set hidden_units default if not provided.
    genome_params_a.setdefault("hidden_units", 16)
    genome_params_b.setdefault("hidden_units", 16)
    genome_params_a.setdefault("activation", "tanh")
    genome_params_b.setdefault("activation", "tanh")
    genome_params_a.setdefault("extra_layer", False)
    genome_params_b.setdefault("extra_layer", False)
    
    # Initialize populations A and B.
    pop_a = [Genome(input_size_a, genome_params_a["hidden_units"], DEFAULT_OUTPUT_UNITS,
                    genome_params_a["activation"], genome_params_a["extra_layer"],
                    genome_params_a["img_size"]) for _ in range(pop_size // 2)]
    pop_b = [Genome(input_size_b, genome_params_b["hidden_units"], DEFAULT_OUTPUT_UNITS,
                    genome_params_b["activation"], genome_params_b["extra_layer"],
                    genome_params_b["img_size"]) for _ in range(pop_size // 2)]
    
    # Lists for tracking metrics.
    avg_rewards_a_history = []
    avg_rewards_b_history = []
    best_fitness_a_history = []
    best_fitness_b_history = []
    
    # Evolution loop.
    for gen in range(generations):
        print(f"\n--- {name}: Generation {gen+1} ---")
        avg_reward_a, avg_reward_b = evaluate_population(pop_a, pop_b)
        avg_rewards_a_history.append(avg_reward_a)
        avg_rewards_b_history.append(avg_reward_b)
        best_fitness_a = max(g.fitness for g in pop_a)
        best_fitness_b = max(g.fitness for g in pop_b)
        best_fitness_a_history.append(best_fitness_a)
        best_fitness_b_history.append(best_fitness_b)
        
        print(f"Generation {gen+1} | Avg Reward A: {avg_reward_a:.2f}, Avg Reward B: {avg_reward_b:.2f}")
        print(f"Generation {gen+1} | Best Fitness A: {best_fitness_a:.2f}, Best Fitness B: {best_fitness_b:.2f}")
        
        # Save best genomes for both populations.
        best_genome_a = max(pop_a, key=lambda g: g.fitness)
        best_genome_b = max(pop_b, key=lambda g: g.fitness)
        best_genome_a.save_weights(os.path.join(weights_dir, f"gen{gen+1}_popA.npz"))
        best_genome_b.save_weights(os.path.join(weights_dir, f"gen{gen+1}_popB.npz"))
        
        # Selection and reproduction; freeze according to configuration.
        pop_a = select_and_reproduce(pop_a, freeze == "a", elite_k, mutation_rate)
        pop_b = select_and_reproduce(pop_b, freeze == "b", elite_k, mutation_rate)
    
    # Save metrics to file.
    metrics_path = os.path.join(results_dir, "metrics.npz")
    np.savez(metrics_path,
             avg_rewards_a=np.array(avg_rewards_a_history),
             avg_rewards_b=np.array(avg_rewards_b_history),
             best_fitness_a=np.array(best_fitness_a_history),
             best_fitness_b=np.array(best_fitness_b_history))
    
    # Visualization: Plot average rewards and best fitness over generations.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards_a_history, label="Avg Reward - Agent A")
    plt.plot(avg_rewards_b_history, label="Avg Reward - Agent B")
    plt.xlabel("Generation")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards Over Generations")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(best_fitness_a_history, label="Best Fitness - Agent A")
    plt.plot(best_fitness_b_history, label="Best Fitness - Agent B")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Experiment '{name}' finished. Metrics and weights saved in '{results_dir}'.")


# -----------------------
# Run Multiple Experiments
# -----------------------
def run_experiments():
    """
    Defines and runs a series of experiments:
      1. Policy Configuration Exploration (baseline, ReLU, extra layer).
      2. Training Dynamics with Competing Policies (asymmetric policies).
      3. Frozen Parameter Experiments (freeze pop A and pop B).
      4. Cross-Generational Benchmarking (implicitly via weight saving and metrics tracking).
    """
    experiments = [
        # Experiment 1: Policy Configuration Exploration - Baseline (two-layer, tanh)
        {
            "name": "Policy_Config_Baseline",
            "pop_size": 20,
            "generations": 50,
            "elite_k": 4,
            "mutation_rate": 0.9,
            "freeze": None,
            "genome_params_a": {"hidden_units": 16, "activation": "tanh", "extra_layer": False, "img_size": (20,20)},
        },
        # Experiment 1 variant: Using ReLU activation.
        {
            "name": "Policy_Config_ReLU",
            "pop_size": 20,
            "generations": 50,
            "elite_k": 4,
            "mutation_rate": 0.9,
            "freeze": None,
            "genome_params_a": {"hidden_units": 16, "activation": "relu", "extra_layer": False, "img_size": (20,20)},
        },
        # Experiment 1 variant: Extra Layer (three-layer network)
        {
            "name": "Policy_Config_ExtraLayer",
            "pop_size": 20,
            "generations": 50,
            "elite_k": 4,
            "mutation_rate": 0.9,
            "freeze": None,
            "genome_params_a": {"hidden_units": 16, "activation": "tanh", "extra_layer": True, "img_size": (20,20)},
        },
        # Experiment 2: Asymmetric Policies (Population A baseline vs. Population B with ReLU)
        {
            "name": "Asymmetric_Policies",
            "pop_size": 20,
            "generations": 50,
            "elite_k": 4,
            "mutation_rate": 0.9,
            "freeze": None,
            "genome_params_a": {"hidden_units": 16, "activation": "tanh", "extra_layer": False, "img_size": (20,20)},
            "genome_params_b": {"hidden_units": 16, "activation": "relu", "extra_layer": False, "img_size": (20,20)},
        },
        # Experiment 3: Frozen Parameter Experiment - Freeze Population A
        {
            "name": "Freeze_Population_A",
            "pop_size": 20,
            "generations": 50,
            "elite_k": 4,
            "mutation_rate": 0.9,
            "freeze": "a",  # Freeze pop A; only pop B evolves.
            "genome_params_a": {"hidden_units": 16, "activation": "tanh", "extra_layer": False, "img_size": (20,20)},
        },
        # Experiment 3 variant: Freeze Population B
        {
            "name": "Freeze_Population_B",
            "pop_size": 20,
            "generations": 50,
            "elite_k": 4,
            "mutation_rate": 0.9,
            "freeze": "b",  # Freeze pop B; only pop A evolves.
            "genome_params_a": {"hidden_units": 16, "activation": "tanh", "extra_layer": False, "img_size": (20,20)},
        },
    ]
    
    for exp_config in experiments:
        evolution_session(exp_config)

# -----------------------
# Main Entry Point
# ----------------------- 
if __name__ == '__main__':
    run_experiments()