# Environment Setup

I use `uv` for dependency management. If you're curious, read the docs [here](https://docs.astral.sh/uv/). If you want to follow the exact expected setup steps, install `uv`. Depending on what platform you are on, the sync process might be different. For CPU-based workflows (MacOS, I see you), you should run:

```
uv sync --extra cpu
```

For CUDA-enabled devices, run

```
uv sync --extra cu124
```

I also have cu121 for older GPUs configured. Just replace the extra term. Then, activate the environment:

```
.venv/Scripts/activate # Windows
source .venv/bin/activate # Linux
```

If you're in VSCode, make sure to select this .venv Python environment as the workspace Python environment when prompted. I also use pre-commit in the development workflow. Once everything is installed, you should run

```
uv run pre-commit install # omit uv run if you're already in the virtual environment
```

# pip Setup

I leave a `requirements.txt` file here for those who do not want to use `uv`. Simply create your own 3.10 Python virtual environment, activate it and run

```
pip install -r requirements.txt
```

to install the necessary dependencies. You can use different Python versions and/or dependency manager, but I cannot guarantee that your behaviors will be similar to mine when running the same scripts.

The rest should be similar to the instructions above.

# devcontainer Setup

If you're having trouble with either, I've also included a devcontainer setup. Install Docker and the Remote Container extension in VSCode, then F1 -> Open Folder in devcontainer. Everything should be configured for you, including `uv` and useful extensions. When everything is done, simply run

```
source .venv/bin/activate
```

to activate and work in the virtual environment.