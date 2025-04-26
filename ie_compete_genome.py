#!/usr/bin/env python3
import argparse
import pickle
import neat
import numpy as np
import csv
from main import simulate_match_net
from loguru import logger
import multiprocessing
from multiprocessing import cpu_count


def _eval_round(task):
    # Unpack task parameters
    idx, config_path, genome_a_path, genome_b_path, img_size, penalty_reward, hit_reward = task
    # Load config and genomes
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    genomeA = load_genome(genome_a_path)
    genomeB = load_genome(genome_b_path)
    # Create networks and run simulation
    netA = neat.nn.FeedForwardNetwork.create(genomeA, config)
    netB = neat.nn.FeedForwardNetwork.create(genomeB, config)
    scoreA, scoreB = simulate_match_net(
        netA, netB, blank_side=None, img_size=img_size,
        penalty_reward=penalty_reward, hit_reward=hit_reward
    )
    return idx, scoreA, scoreB


def load_genome(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare two NEAT genomes over multiple rounds of Boxing matches")
    parser.add_argument('--config', type=str, default='config.ini', help='Path to NEAT config file')
    parser.add_argument('--genome_a', type=str, required=True, help='Path to genome pickle for agent A')
    parser.add_argument('--genome_b', type=str, required=True, help='Path to genome pickle for agent B')
    parser.add_argument('--img_size', type=int, nargs=2, default=[20,20], help='Preprocessing resolution (width height)')
    parser.add_argument('--rounds', type=int, default=10, help='Number of matches to run')
    parser.add_argument('--penalty_reward', type=float, default=None, help='Override negative rewards')
    parser.add_argument('--hit_reward', type=float, default=None, help='Override positive rewards')
    parser.add_argument('--out_csv', type=str, default='results/compare_results.csv', help='Output CSV path')
    args = parser.parse_args()

    img_size = tuple(args.img_size)

    # Prepare tasks for parallel processing
    tasks = [(i, args.config, args.genome_a, args.genome_b, img_size, args.penalty_reward, args.hit_reward) for i in range(1, args.rounds + 1)]

    # Create a pool of workers and run the tasks
    with multiprocessing.Pool(processes=cpu_count()) as pool:
        results = pool.map(_eval_round, tasks)

    # Log results
    for i, scoreA, scoreB in results:
        logger.info(f"Round {i}: A={scoreA:.2f}, B={scoreB:.2f}")

    # save CSV
    with open(args.out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['round', 'scoreA', 'scoreB'])
        writer.writerows(results)
    logger.info(f"Results written to {args.out_csv}")

    # summary statistics
    scoresA = np.array([r[1] for r in results])
    scoresB = np.array([r[2] for r in results])
    logger.info(f"Agent A average: {scoresA.mean():.2f}, stddev: {scoresA.std():.2f}")
    logger.info(f"Agent B average: {scoresB.mean():.2f}, stddev: {scoresB.std():.2f}")

    # save summary statistics to text file
    summary_file = args.out_csv.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Agent A average: {scoresA.mean():.2f}, stddev: {scoresA.std():.2f}\n")
        f.write(f"Agent B average: {scoresB.mean():.2f}, stddev: {scoresB.std():.2f}\n")
    logger.info(f"Summary statistics written to {summary_file}")
    logger.info("Done!")

if __name__ == '__main__':
    main()
