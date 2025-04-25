!/bin/bash

# Baseline experiment (RUNNING)
uv run python interleaved_evolution.py --config neat_config_boxing.ini --log-level INFO --img_size 20 20 --generations 50 --pop_size 20 --results results/baseline

# Blank b experiment (RUNNING)
uv run python interleaved_evolution.py --config neat_config_boxing.ini --log-level INFO --img_size 20 20 --generations 50 --pop_size 20 --results results/blank_b --blank_size b

# Long generation experiment (RUNNING)
uv run python interleaved_evolution.py --config neat_config_boxing.ini --log-level INFO --img_size 20 20 --generations 200 --pop_size 20 --results results/long_generation

# Long generation with blank b experiment
uv run python interleaved_evolution.py --config neat_config_boxing.ini --log-level INFO --img_size 20 20 --generations 200 --pop_size 20 --results results/long_generation_blank_b --blank_size b

# No penalty experiment
uv run python interleaved_evolution.py --config neat_config_boxing.ini --log-level INFO --img_size 20 20 --generations 50 --pop_size 20 --results results/no_penalty --penalty_reward 0.0

# Half penalty experiment
uv run python interleaved_evolution.py --config neat_config_boxing.ini --log-level INFO --img_size 20 20 --generations 50 --pop_size 20 --results results/half_penalty --penalty_reward -0.5

# Inverse penalty experiment
uv run python interleaved_evolution.py --config neat_config_boxing.ini --log-level INFO --img_size 20 20 --generations 50 --pop_size 20 --results results/inverse_penalty --penalty_reward 1.0 --hit_reward -1.0