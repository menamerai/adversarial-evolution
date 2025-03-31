from pettingzoo.atari import boxing_v2
import numpy as np
import random
import uuid
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- Global Parameters ---
POP_SIZE = 20
GENERATIONS = 50
ELITE_K = 4
MUTATION_RATE = 0.9

# --- Genome Definition ---
class Genome:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.w1 = np.random.randn(400, 16) * 0.5
        self.b1 = np.zeros(16)
        self.w2 = np.random.randn(16, 18) * 0.5
        self.b2 = np.zeros(18)
        self.fitness = 0

    def mutate(self):
        for param in [self.w1, self.b1, self.w2, self.b2]:
            param += np.random.randn(*param.shape) * 0.05

    def clone(self):
        child = Genome()
        child.w1 = np.copy(self.w1)
        child.b1 = np.copy(self.b1)
        child.w2 = np.copy(self.w2)
        child.b2 = np.copy(self.b2)
        return child

def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (20, 20))
    normalized = resized / 255.0
    return normalized.flatten()

# --- Genome Policy ---
def genome_policy(observation: np.ndarray, genome: Genome) -> int:
    x = preprocess(observation)
    h = np.tanh(np.dot(x, genome.w1) + genome.b1)
    logits = np.dot(h, genome.w2) + genome.b2
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return np.random.choice(len(probs), p=probs)

# --- Simulate Match ---
def simulate_match(genome_a, genome_b) -> Tuple[int, int]:
    env = boxing_v2.parallel_env(render_mode=None)
    obs, _ = env.reset()
    agents = env.agents

    total_reward_a, total_reward_b = 0, 0
    step_count = 0

    while env.agents:
        actions = {
            agents[0]: genome_policy(obs[agents[0]], genome_a),
            agents[1]: genome_policy(obs[agents[1]], genome_b),
        }
        obs, rewards, terms, truncs, _ = env.step(actions)

        total_reward_a += rewards.get(agents[0], 0)
        total_reward_b += rewards.get(agents[1], 0)
        step_count += 1

        if all(terms.values()) or all(truncs.values()):
            break

    env.close()
    return total_reward_a, total_reward_b

# --- Evaluate Population ---
def evaluate_population(pop_a: List[Genome], pop_b: List[Genome]):
    for genome in pop_a + pop_b:
        genome.fitness = 0  # Reset fitness before evaluation

    total_rewards_a, total_rewards_b = 0, 0

    for genome_a, genome_b in zip(pop_a, pop_b):
        reward_a, reward_b = simulate_match(genome_a, genome_b)
        genome_a.fitness += reward_a
        genome_b.fitness += reward_b

        total_rewards_a += reward_a
        total_rewards_b += reward_b

    avg_reward_a = total_rewards_a / len(pop_a)
    avg_reward_b = total_rewards_b / len(pop_b)

    return avg_reward_a, avg_reward_b

# --- Selection and Reproduction ---
def select_and_reproduce(pop: List[Genome]) -> List[Genome]:
    pop.sort(key=lambda g: g.fitness, reverse=True)
    elites = pop[:ELITE_K]
    new_pop = []
    while len(new_pop) < len(pop):
        parent = random.choice(elites)
        child = parent.clone()
        if random.random() < MUTATION_RATE:
            child.mutate()
        new_pop.append(child)
    return new_pop

# --- Main Evolution Loop ---
population = [Genome() for _ in range(POP_SIZE)]
pop_a, pop_b = population[:POP_SIZE//2], population[POP_SIZE//2:]

avg_rewards_a_history, avg_rewards_b_history = [], []

for gen in range(GENERATIONS):
    print(f"\n--- Starting Generation {gen+1} ---")

    avg_reward_a, avg_reward_b = evaluate_population(pop_a, pop_b)
    avg_rewards_a_history.append(avg_reward_a)
    avg_rewards_b_history.append(avg_reward_b)

    top_fitness = max(g.fitness for g in population)
    avg_fitness = np.mean([g.fitness for g in population])
    print(f"Generation {gen+1} | Top Fitness: {top_fitness}, Avg Fitness: {avg_fitness:.2f}")
    print(f"Average Reward - Agent A: {avg_reward_a}, Agent B: {avg_reward_b}")

    pop_a = select_and_reproduce(pop_a)
    pop_b = select_and_reproduce(pop_b)
    population = pop_a + pop_b

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(avg_rewards_a_history, label="Average Reward - Agent A")
plt.plot(avg_rewards_b_history, label="Average Reward - Agent B")
plt.xlabel("Generation")
plt.ylabel("Average Reward")
plt.title("Evolution of Average Rewards Over Generations")
plt.legend()
plt.grid(True)
plt.show()