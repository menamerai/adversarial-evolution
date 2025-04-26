#!/usr/bin/env python3
import os
import argparse
import pickle
import neat
import cv2
import sys
import random
import numpy as np
from pettingzoo.atari import boxing_v2
from pettingzoo.utils.wrappers import BaseParallelWrapper
from loguru import logger
import time
import matplotlib.pyplot as plt
import itertools
import multiprocessing
from multiprocessing import cpu_count
from functools import partial

# Set up loguru logger
logger.remove()  # Remove default handler
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="INFO",
)


# wrapper for modifying rewards
class RewardModifier(BaseParallelWrapper):
    """
    Wraps a parallel env and overrides negative rewards with a fixed penalty_reward.
    If penalty_reward is None, leaves rewards untouched.
    """

    def __init__(self, env, penalty_reward: float = None, hit_reward: float = None):
        super().__init__(env)
        self.penalty_reward = penalty_reward
        self.hit_reward = hit_reward

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        if self.penalty_reward is not None or self.hit_reward is not None:
            for agent, r in rewards.items():
                if r < 0 and self.penalty_reward is not None:
                    rewards[agent] = self.penalty_reward
                elif r > 0 and self.hit_reward is not None:
                    rewards[agent] = self.hit_reward
        return obs, rewards, terms, truncs, infos


# -----------------------
# Preprocessing and Policy
# -----------------------
def preprocess(obs: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Convert RGB observation to grayscale, resize, normalize, and flatten.
    """
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, img_size)
    return resized.flatten() / 255.0


def get_action(
    net: neat.nn.FeedForwardNetwork, obs: np.ndarray, img_size: tuple
) -> int:
    """
    Given a NEAT network and raw observation, return an action via softmax sampling.
    """
    x = preprocess(obs, img_size)
    out = net.activate(x)
    exp_out = np.exp(out - np.max(out))
    probs = exp_out / np.sum(exp_out)
    return int(np.random.choice(len(probs), p=probs))


# -----------------------
# Simulation
# -----------------------
def simulate_match_net(
    netA: neat.nn.FeedForwardNetwork,
    netB: neat.nn.FeedForwardNetwork,
    blank_side: str,
    img_size: tuple,
    penalty_reward: float = None,
    hit_reward: float = None,
) -> tuple:
    """
    Play one parallel Boxing match between two NEAT networks, optionally blanking one side (no-op).
    Returns (rewardA, rewardB).
    """
    logger.debug("Starting new match simulation")
    start_time = time.time()

    # Environment setup
    logger.trace("Initializing boxing environment")
    env_start = time.time()
    base_env = boxing_v2.parallel_env(render_mode=None)
    # wrap to modify penalties
    env = RewardModifier(base_env, penalty_reward=penalty_reward, hit_reward=hit_reward)
    logger.trace(f"Environment initialization took {time.time() - env_start:.4f}s")

    # Reset environment
    reset_start = time.time()
    obs, _ = env.reset()
    logger.trace(f"Environment reset took {time.time() - reset_start:.4f}s")

    # Store agent IDs at the beginning since they'll remain consistent
    if len(env.agents) < 2:
        logger.error(
            f"Expected 2 agents, but environment has {len(env.agents)}: {env.agents}"
        )
        return 0, 0

    agent_a = env.agents[0]
    agent_b = env.agents[1]
    logger.trace(f"Agent A ID: {agent_a}, Agent B ID: {agent_b}")

    total_a, total_b = 0, 0
    steps = 0
    action_time_a = 0
    action_time_b = 0
    step_time = 0

    while env.agents:
        steps += 1
        step_start = time.time()

        # Safe handling of actions
        actions = {}

        # Determine actions, blanking side if requested
        if blank_side == "a":
            actionA = 0  # NOOP
            logger.trace(f"Step {steps}: Agent A using NOOP (blank side)")
        else:
            if agent_a in obs:
                action_start = time.time()
                actionA = get_action(netA, obs[agent_a], img_size)
                action_time = time.time() - action_start
                action_time_a += action_time
                logger.trace(
                    f"Step {steps}: Agent A action selection took {action_time:.4f}s - action: {actionA}"
                )
            else:
                logger.trace(
                    f"Step {steps}: Agent A not in observation, skipping action"
                )
                actionA = 0

        if blank_side == "b":
            actionB = 0
            logger.trace(f"Step {steps}: Agent B using NOOP (blank side)")
        else:
            if agent_b in obs:
                action_start = time.time()
                actionB = get_action(netB, obs[agent_b], img_size)
                action_time = time.time() - action_start
                action_time_b += action_time
                logger.trace(
                    f"Step {steps}: Agent B action selection took {action_time:.4f}s - action: {actionB}"
                )
            else:
                logger.trace(
                    f"Step {steps}: Agent B not in observation, skipping action"
                )
                actionB = 0

        # Only add actions for agents that are still active
        if agent_a in env.agents:
            actions[agent_a] = actionA
        if agent_b in env.agents:
            actions[agent_b] = actionB

        # Log actions being taken
        logger.trace(f"Step {steps}: Actions: {actions}")

        # Environment step
        if not actions:
            logger.warning(f"Step {steps}: No active agents remain, breaking loop")
            break

        env_step_start = time.time()
        obs, rewards, terms, truncs, _ = env.step(actions)
        env_step_time = time.time() - env_step_start
        step_time += env_step_time
        logger.trace(f"Step {steps}: Environment step took {env_step_time:.4f}s")
        logger.trace(
            f"Step {steps}: Rewards: {rewards}, Terms: {terms}, Truncs: {truncs}"
        )

        # Reward tracking (safely)
        total_a += float(rewards.get(agent_a, 0))
        total_b += float(rewards.get(agent_b, 0))

        # Periodic logging
        if steps % 10 == 0:  # Increased frequency from 50 to 10 for better visibility
            current_time = time.time() - start_time
            logger.trace(
                f"Step {steps}: A({total_a}) vs B({total_b}) - Time elapsed: {current_time:.2f}s, Avg step: {current_time/steps:.4f}s"
            )
            logger.trace(f"Step {steps}: Agents remaining: {env.agents}")

        if all(terms.values()) or all(truncs.values()) or not env.agents:
            logger.trace(
                f"Step {steps}: Match terminated - terms: {terms}, truncs: {truncs}, agents: {env.agents}"
            )
            break

    # Close environment and report stats
    env.close()
    total_time = time.time() - start_time

    # Performance analysis
    logger.debug(f"Match complete after {steps} steps: A({total_a}) vs B({total_b})")
    logger.debug(
        f"Match duration: {total_time:.2f}s, Average time per step: {total_time/steps:.4f}s"
    )
    logger.debug(
        f"Agent A total action time: {action_time_a:.2f}s, avg: {action_time_a/steps:.4f}s"
    )
    logger.debug(
        f"Agent B total action time: {action_time_b:.2f}s, avg: {action_time_b/steps:.4f}s"
    )
    logger.debug(
        f"Environment step total time: {step_time:.2f}s, avg: {step_time/steps:.4f}s"
    )

    # Identify bottlenecks
    total_compute = action_time_a + action_time_b + step_time
    overhead = total_time - total_compute
    logger.debug(
        f"Overhead time: {overhead:.2f}s ({100*overhead/total_time:.1f}% of total)"
    )

    return total_a, total_b


# Parallel event worker for coevolution
def _eval_event(event, configA, configB, blank_side, img_size, penalty_reward, hit_reward):
    """
    Worker to evaluate a single coevolution event:
    event = ("A"/"B", gid, genome, opp_gid, opp_genome)
    Returns (label, gid, opp_gid, fitA, fitB).
    """
    label, gid, genome, opp_gid, opp_genome = event
    # Silence detailed logging for performance
    logger_disabled = logger.level("TRACE")[0]  # store current level index
    logger.disable("all")
    try:
        if label == "A":
            netA = neat.nn.FeedForwardNetwork.create(genome, configA)
            netB = neat.nn.FeedForwardNetwork.create(opp_genome, configB)
            fitA, fitB = simulate_match_net(
                netA, netB, blank_side, img_size, penalty_reward, hit_reward)
        else:
            netA = neat.nn.FeedForwardNetwork.create(opp_genome, configA)
            netB = neat.nn.FeedForwardNetwork.create(genome, configB)
            fitA, fitB = simulate_match_net(
                netA, netB, blank_side, img_size, penalty_reward, hit_reward)
    finally:
        logger.enable("all")
    return label, gid, opp_gid, fitA, fitB


# -----------------------
# Coevolution with NEAT
# -----------------------
def coevolve(
    config_file: str,
    pop_size: int,
    generations: int,
    mutation_rate: float,
    blank_side: str,
    unblank_gen: int,
    penalty_reward: float,
    hit_reward: float,
    img_size: tuple,
    results_dir: str,
):
    import os
    # ensure config_file points to the script’s directory
    if not os.path.isabs(config_file):
        config_file = os.path.join(os.path.dirname(__file__), config_file)
        
    logger.info(f"Starting coevolution with populations of size {pop_size}")
    logger.info(f"Generations: {generations}, Mutation rate: {mutation_rate}")
    logger.info(f"Image size: {img_size}, Blank side: {blank_side}")

    # Load NEAT configurations for two separate populations
    logger.info(f"Loading NEAT config from {config_file}")
    try:
        configA = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )
        configB = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )
        logger.success("NEAT config loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

    # Override hyperparameters programmatically
    configA.pop_size = pop_size
    configB.pop_size = pop_size
    configA.genome_config.weight_mutate_rate = mutation_rate
    configB.genome_config.weight_mutate_rate = mutation_rate
    logger.debug("Applied custom hyperparameters to config")

    # Initialize populations
    popA = neat.Population(configA)
    popB = neat.Population(configB)
    logger.info("Initialized populations A and B")

    # Reporters for logging
    popA.add_reporter(neat.StdOutReporter(True))
    statsA = neat.StatisticsReporter()
    popA.add_reporter(statsA)
    popB.add_reporter(neat.StdOutReporter(False))
    statsB = neat.StatisticsReporter()
    popB.add_reporter(statsB)
    logger.debug("Added reporters to populations")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # For tracking fitness over generations
    gen_numbers = []
    avg_fitness_A = []
    best_fitness_A = []
    avg_fitness_B = []
    best_fitness_B = []

    # Interleaved coevolutionary loop
    for gen in range(1, generations+1):
        logger.info(f"=== Generation {gen}/{generations} (interleaved) ===")

        # determine whether the original blank_side is still in effect
        if unblank_gen is not None and gen >= unblank_gen:
            if blank_side is not None:
                logger.info(f"Gen {gen}: unblanking side '{blank_side}', both sides now act")
            effective_blank = None
        else:
            effective_blank = blank_side

        # decide if we’re still in single‑pop mode
        effective_blank = None if (unblank_gen is not None and gen >= unblank_gen) else blank_side
        single_mode = blank_side is not None and unblank_gen is not None and gen < unblank_gen
        if single_mode:
            # which side is active?
            active = 'B' if blank_side == 'a' else 'A'
            logger.info(f"Gen {gen}: single‑population evolution for {active} only")
            # bind active/other pops & configs
            if active == 'A':
                pop_active, pop_other = popA, popB
                cfg_active, _ = configA, configB
                label = 'A'
            else:
                pop_active, pop_other = popB, popA
                cfg_active, _ = configB, configA
                label = 'B'
            # zero fitness on active
            items = list(pop_active.population.items())
            for _, g in items:
                g.fitness = 0.0
            # build single‑side events
            events = []
            for gid, genome in items:
                opp_gid, opp_genome = random.choice(list(pop_other.population.items()))
                events.append((label, gid, genome, opp_gid, opp_genome))
            random.shuffle(events)
            # evaluate
            with multiprocessing.Pool(cpu_count()) as pool:
                func = partial(
                    _eval_event,
                    configA=configA,
                    configB=configB,
                    blank_side=effective_blank,
                    img_size=img_size,
                    penalty_reward=penalty_reward,
                    hit_reward=hit_reward,
                )
                results = pool.map(func, events)
            # assign & normalize fitness on active only
            for lab, gid, _, fitA, fitB in results:
                pop_active.population[gid].fitness += (fitA if lab == 'A' else fitB)
            if events:
                for _, g in items:
                    g.fitness /= len(events)
            # reproduce & speciate active only
            pop_active.population = pop_active.reproduction.reproduce(
                cfg_active, pop_active.species, cfg_active.pop_size, gen
            )
            pop_active.species.speciate(cfg_active, pop_active.population, gen)
            # record stats: active gets real, inactive padded zero (filter out None)
            gen_numbers.append(gen)
            if active == 'A':
                vals = [g.fitness for g in popA.population.values() if g.fitness is not None]
                if vals:
                    avg_fitness_A.append(sum(vals)/len(vals))
                    best_fitness_A.append(max(vals))
                else:
                    avg_fitness_A.append(0); best_fitness_A.append(0)
                avg_fitness_B.append(0); best_fitness_B.append(0)
            else:
                avg_fitness_A.append(0); best_fitness_A.append(0)
                vals = [g.fitness for g in popB.population.values() if g.fitness is not None]
                if vals:
                    avg_fitness_B.append(sum(vals)/len(vals))
                    best_fitness_B.append(max(vals))
                else:
                    avg_fitness_B.append(0); best_fitness_B.append(0)
            continue

        # Grab current genomes
        itemsA = list(popA.population.items())  # [(gid, genome), …]
        itemsB = list(popB.population.items())

        # Zero–out fitness before we begin
        for _, g in itemsA + itemsB:
            g.fitness = 0.0

        # Prepare interleaved events list
        events = []
        for gid, genome in itemsA:
            opp_gid, opp_genome = random.choice(itemsB)
            events.append(("A", gid, genome, opp_gid, opp_genome))
        for gid, genome in itemsB:
            opp_gid, opp_genome = random.choice(itemsA)
            events.append(("B", gid, genome, opp_gid, opp_genome))

        # Shuffle events to randomize workload
        random.shuffle(events)

        logger.info(f"Interleaved evaluation event count: {len(events)} (blank_side={effective_blank})")

        # Parallel evaluation of events
        with multiprocessing.Pool(cpu_count()) as pool:
            func = partial(
                _eval_event,
                configA=configA,
                configB=configB,
                blank_side=effective_blank,
                img_size=img_size,
                penalty_reward=penalty_reward,
                hit_reward=hit_reward,
            )
            results = pool.map(func, events)

        # Aggregate fitness assignments
        for label, gid, opp_gid, fitA, fitB in results:
            if label == "A":
                popA.population[gid].fitness += fitA
                popB.population[opp_gid].fitness += fitB
            else:
                popB.population[gid].fitness += fitB
                popA.population[opp_gid].fitness += fitA

        # (Optional) Normalize fitness by number of matches each got:
        matches_per = len([1 for lab, *_ in events if lab == "A" and effective_blank != "a"])
        if matches_per:
            logger.debug(
                f"[Gen {gen}] normalizing A fitness over {matches_per} matches (blanked? {effective_blank=='a'})"
            )
            for _, g in itemsA:
                g.fitness /= matches_per
        matches_per = len([1 for lab, *_ in events if lab == "B" and effective_blank != "b"])
        if matches_per:
            logger.debug(
                f"[Gen {gen}] normalizing B fitness over {matches_per} matches (blanked? {effective_blank=='b'})"
            )
            for _, g in itemsB:
                g.fitness /= matches_per

        # Now do reproduction on both populations
        logger.info(f"[Gen {gen}] reproducing populations A and B")
        popA.population = popA.reproduction.reproduce(
            configA, popA.species, configA.pop_size, gen
        )
        popB.population = popB.reproduction.reproduce(
            configB, popB.species, configB.pop_size, gen
        )
        popA.species.speciate(configA, popA.population, gen)
        popB.species.speciate(configB, popB.population, gen)
        logger.success(
            f"[Gen {gen}] interleaved evolution complete. "
            f"Species A: {len(popA.species.species)}, "
            f"Species B: {len(popB.species.species)}"
        )

        # Save the best genomes each generation
        try:
            # Filter out genomes with None fitness
            valid_genomesA = [
                g for g in popA.population.values() if g.fitness is not None
            ]
            valid_genomesB = [
                g for g in popB.population.values() if g.fitness is not None
            ]

            # Calculate and store statistics for plotting
            gen_numbers.append(gen)

            if valid_genomesA and effective_blank != "a":
                bestA = max(valid_genomesA, key=lambda g: g.fitness)
                avg_fit_A = sum(g.fitness for g in valid_genomesA) / len(valid_genomesA)
                best_fitness_A.append(bestA.fitness)
                avg_fitness_A.append(avg_fit_A)

                bestA_path = os.path.join(results_dir, f"best_gen{gen}_A.pkl")
                with open(bestA_path, "wb") as fa:
                    pickle.dump(bestA, fa)
                logger.info(
                    f"Gen {gen} best fitness A: {bestA.fitness:.2f}, avg: {avg_fit_A:.2f}"
                )
                logger.info(f"Saved best A genome to {bestA_path}")
            else:
                logger.warning(
                    f"Gen {gen}: No valid fitness values found in population A"
                )
                best_fitness_A.append(0)
                avg_fitness_A.append(0)

            if valid_genomesB and effective_blank != "b":
                bestB = max(valid_genomesB, key=lambda g: g.fitness)
                avg_fit_B = sum(g.fitness for g in valid_genomesB) / len(valid_genomesB)
                best_fitness_B.append(bestB.fitness)
                avg_fitness_B.append(avg_fit_B)

                bestB_path = os.path.join(results_dir, f"best_gen{gen}_B.pkl")
                with open(bestB_path, "wb") as fb:
                    pickle.dump(bestB, fb)
                logger.info(
                    f"Gen {gen} best fitness B: {bestB.fitness:.2f}, avg: {avg_fit_B:.2f}"
                )
                logger.info(f"Saved best B genome to {bestB_path}")
            else:
                logger.warning(
                    f"Gen {gen}: No valid fitness values found in population B"
                )
                best_fitness_B.append(0)
                avg_fitness_B.append(0)

            # Log species info
            logger.info(f"Population A has {len(popA.species.species)} species")
            logger.info(f"Population B has {len(popB.species.species)} species")

            # Generate and save plots every 5 generations
            if gen % 5 == 0 or gen == generations:
                plot_fitness_curves(
                    gen_numbers,
                    avg_fitness_A,
                    best_fitness_A,
                    avg_fitness_B,
                    best_fitness_B,
                    results_dir,
                )

        except Exception as e:
            logger.error(f"Error processing generation {gen} results: {str(e)}")

    # Final plots
    plot_fitness_curves(
        gen_numbers,
        avg_fitness_A,
        best_fitness_A,
        avg_fitness_B,
        best_fitness_B,
        results_dir,
    )

    logger.success(f"Coevolution complete after {generations} generations")


def plot_fitness_curves(
    gen_numbers,
    avg_fitness_A,
    best_fitness_A,
    avg_fitness_B,
    best_fitness_B,
    results_dir,
):
    """
    Plot and save fitness curves showing both average and best fitness
    for populations A and B across generations.
    """
    try:
        # Regular plot
        plt.figure(figsize=(12, 8))

        # Create subplot for Population A
        plt.subplot(2, 1, 1)
        plt.plot(gen_numbers, best_fitness_A, "b-", label="Best Fitness")
        plt.plot(gen_numbers, avg_fitness_A, "b--", label="Average Fitness")
        plt.title("Population A Fitness Over Generations")
        plt.ylabel("Fitness")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Create subplot for Population B
        plt.subplot(2, 1, 2)
        plt.plot(gen_numbers, best_fitness_B, "r-", label="Best Fitness")
        plt.plot(gen_numbers, avg_fitness_B, "r--", label="Average Fitness")
        plt.title("Population B Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        plt.tight_layout()

        # Save regular plot
        plot_path = os.path.join(results_dir, f"fitness_curves.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Enhanced plot with additional information
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        plt.figure(figsize=(12, 10))

        # Add title with timestamp
        plt.suptitle(f"Training Progress - {current_time}", fontsize=16)

        # Create subplot for Population A
        plt.subplot(2, 2, 1)
        plt.plot(gen_numbers, best_fitness_A, "b-", label="Best Fitness")
        plt.plot(gen_numbers, avg_fitness_A, "b--", label="Average Fitness")
        plt.title("Population A Fitness")
        plt.ylabel("Fitness")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Create subplot for Population B
        plt.subplot(2, 2, 2)
        plt.plot(gen_numbers, best_fitness_B, "r-", label="Best Fitness")
        plt.plot(gen_numbers, avg_fitness_B, "r--", label="Average Fitness")
        plt.title("Population B Fitness")
        plt.ylabel("Fitness")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Relative performance difference
        plt.subplot(2, 2, 3)
        if len(best_fitness_A) > 0 and len(best_fitness_B) > 0:
            # Calculate relative performance (difference between best A and best B)
            performance_diff = [a - b for a, b in zip(best_fitness_A, best_fitness_B)]
            plt.plot(gen_numbers, performance_diff, "g-")
            plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            plt.title("Relative Performance (A - B)")
            plt.xlabel("Generation")
            plt.ylabel("Fitness Difference")
            plt.grid(True, linestyle="--", alpha=0.7)

        # Stats and metadata text box
        plt.subplot(2, 2, 4)
        plt.axis("off")

        # Prepare statistics for display
        stats_text = "Training Statistics:\n\n"

        if len(gen_numbers) > 0:
            current_gen = gen_numbers[-1]
            stats_text += f"Current Generation: {current_gen}\n"

        if len(best_fitness_A) > 0:
            max_a = max(best_fitness_A)
            current_a = best_fitness_A[-1]
            avg_a = sum(best_fitness_A) / len(best_fitness_A)
            stats_text += f"Population A\n"
            stats_text += f"  Current Best: {current_a:.2f}\n"
            stats_text += f"  Max Best: {max_a:.2f}\n"
            stats_text += f"  Avg Best: {avg_a:.2f}\n\n"

        if len(best_fitness_B) > 0:
            max_b = max(best_fitness_B)
            current_b = best_fitness_B[-1]
            avg_b = sum(best_fitness_B) / len(best_fitness_B)
            stats_text += f"Population B\n"
            stats_text += f"  Current Best: {current_b:.2f}\n"
            stats_text += f"  Max Best: {max_b:.2f}\n"
            stats_text += f"  Avg Best: {avg_b:.2f}\n\n"

        # Add improvement rates
        if len(best_fitness_A) > 5:  # Calculate improvement only with enough data
            recent_improvement_a = (
                best_fitness_A[-1] - best_fitness_A[-6]
            )  # Last 5 generations
            stats_text += f"Recent A Improvement: {recent_improvement_a:.2f}\n"

        if len(best_fitness_B) > 5:
            recent_improvement_b = (
                best_fitness_B[-1] - best_fitness_B[-6]
            )  # Last 5 generations
            stats_text += f"Recent B Improvement: {recent_improvement_b:.2f}\n"

        # Add metadata
        stats_text += f"\nTimestamp: {current_time}"

        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

        # Save enhanced plot
        detailed_plot_path = os.path.join(results_dir, f"fitness_detailed.png")
        plt.savefig(detailed_plot_path, dpi=300)
        plt.close()

        logger.info(f"Saved standard fitness plot to {plot_path}")
        logger.info(f"Saved detailed fitness report to {detailed_plot_path}")

    except Exception as e:
        logger.error(f"Error creating fitness plots: {str(e)}")


# -----------------------
# CLI Entry Point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coevolutionary NEAT for Atari Boxing")
    parser.add_argument(
        "--config", type=str, default="neat_config.ini", help="Path to NEAT config file"
    )
    parser.add_argument(
        "--pop_size", type=int, default=20, help="Population size for each species"
    )
    parser.add_argument(
        "--generations", type=int, default=50, help="Number of generations to run"
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.9,
        help="Probability of weight perturbation per genome",
    )
    parser.add_argument(
        "--blank_side",
        choices=["a", "b"],
        default=None,
        help="If specified, that side will perform NOOP each step",
    )
    parser.add_argument(
        "--unblank_gen",
        type=int,
        default=None,
        help="Generation at which to stop blanking the specified side; after this gen both act normally",
    )
    parser.add_argument(
        "--penalty_reward",
        type=float,
        default=None,
        help="Override any negative reward with this value (e.g. 0 to remove penalties)",
    )
    parser.add_argument(
        "--hit_reward",
        type=float,
        default=None,
        help="Override any positive reward with this value",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[20, 20],
        help="Preprocessing resolution (width height)",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results",
        help="Directory to save weights and stats",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    args = parser.parse_args()

    # Update logger with command line log level
    logger.remove()
    logger.add(
        sink=sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        level=args.log_level,
    )

    # Also save logs to file
    log_file = os.path.join(args.results, "evolution.log")
    os.makedirs(args.results, exist_ok=True)
    logger.add(
        sink=log_file,
        rotation="10 MB",
        retention="1 week",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {function}:{line} - {message}",
        level="DEBUG",  # Log everything to file
    )

    logger.info("Boxing Evolution starting")
    logger.info(f"Arguments: {vars(args)}")

    # -------------------
    # Parameter Grid Search
    # -------------------
    grid = {
        'pop_size': [20],
        'mutation_rate': [0.9],
        'penalty_reward': [0.0],
        'hit_reward': [None],
    }

    # Run grid search
    for pop, mut, pen, hit in itertools.product(
        grid['pop_size'],
        grid['mutation_rate'],
        grid['penalty_reward'],
        grid['hit_reward']
    ):
        run_dir = os.path.join(args.results,
                               f"pop{pop}_mut{mut}_pen{pen}_hit{hit}")
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Grid run: pop_size={pop}, mutation_rate={mut}, "
                    f"penalty_reward={pen}, hit_reward={hit}")
        try:
            coevolve(
                config_file=args.config,
                pop_size=pop,
                generations=args.generations,
                mutation_rate=mut,
                blank_side=args.blank_side,
                unblank_gen=args.unblank_gen,
                penalty_reward=pen,
                hit_reward=hit,
                img_size=tuple(args.img_size),
                results_dir=run_dir,
            )
        except Exception as e:
            logger.exception(f"Error during coevolution: {str(e)}")
            raise
