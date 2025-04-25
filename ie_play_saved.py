#!/usr/bin/env python3
import argparse
import pickle
import time

import numpy as np
import cv2
import neat
from pettingzoo.atari import boxing_v2


def load_genome(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess(obs: np.ndarray, img_size: tuple) -> np.ndarray:
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, img_size)
    return resized.flatten() / 255.0


def select_action(net, obs: np.ndarray, img_size: tuple) -> int:
    x = preprocess(obs, img_size)
    out = net.activate(x)
    exp_out = np.exp(out - np.max(out))
    probs = exp_out / np.sum(exp_out)
    return int(np.random.choice(len(probs), p=probs))


def main():
    parser = argparse.ArgumentParser(description="Play saved NEAT genome(s) in Boxing")
    parser.add_argument(
        '--config', type=str, default='config.ini', help='Path to NEAT config file'
    )
    parser.add_argument(
        '--genome_a', type=str, required=True,
        help='Path to genome pickle for agent A'
    )
    parser.add_argument(
        '--genome_b', type=str, default=None,
        help='Path to genome pickle for agent B (optional)'
    )
    parser.add_argument(
        '--img_size', type=int, nargs=2, default=[20,20],
        help='Preprocessing resolution (width height)'
    )
    parser.add_argument(
        '--max_steps', type=int, default=10000,
        help='Maximum number of steps to run'
    )
    parser.add_argument(
        '--target_secs', type=float, default=15.0,
        help='Target video duration in seconds'
    )
    parser.add_argument(
        '--save_mp4', type=str, default=None,
        help='Path to save condensed MP4 (optional)'
    )
    args = parser.parse_args()

    # Error guard for save_mp4 flag
    if args.save_mp4:
        if not args.save_mp4.endswith('.mp4'):
            parser.error("Output file must end with .mp4")
        if args.target_secs <= 0:
            parser.error("target_secs must be positive")

    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )

    # Load genomes and networks
    genomeA = load_genome(args.genome_a)
    netA = neat.nn.FeedForwardNetwork.create(genomeA, config)
    netB = None
    if args.genome_b:
        genomeB = load_genome(args.genome_b)
        netB = neat.nn.FeedForwardNetwork.create(genomeB, config)

    img_size = tuple(args.img_size)

    # Initialize environment in human-render mode
    env = boxing_v2.parallel_env(render_mode='rgb_array' if args.save_mp4 else 'human')
    obs, _ = env.reset()
    if args.save_mp4:
        frames = [env.render()]
    agent_names = env.agents
    agent_a = agent_names[0]
    agent_b = agent_names[1]

    for step in range(args.max_steps):
        actions = {}
        # Agent A action
        if agent_a in obs:
            actions[agent_a] = select_action(netA, obs[agent_a], img_size)
        # Agent B action (random or genome)
        if agent_b in obs:
            if netB:
                actions[agent_b] = select_action(netB, obs[agent_b], img_size)
            else:
                actions[agent_b] = 0  # NOOP
        obs, rewards, terms, truncs, _ = env.step(actions)
        # Slow down for visibility
        time.sleep(0.02)
        if args.save_mp4:
            frames.append(env.render())
        if all(terms.values()) or all(truncs.values()):
            break

    env.close()
    if args.save_mp4:
        height, width, _ = frames[0].shape
        fps = max(1.0, len(frames) / args.target_secs)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save_mp4, fourcc, fps, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saved video to {args.save_mp4}")


if __name__ == '__main__':
    main()