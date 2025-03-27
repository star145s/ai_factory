# Validator

Validators download the models from hugging face for each miner based on the Bittensor chain metadata and continuously evaluate them, setting weights based on the performance of each model against a dataset for each competition. They also log results to [wandb](https://wandb.ai/ai-factory/ai-factory-validators).

# System Requirements

Validators will need enough disk space to store the models of miners being evaluated. Each model has a max size by block defined in [constants/**init**.py] and the validator has cleanup logic to remove old models. It is recommended to have at least 2 TB of disk space and 80GB of system memory.

# Getting Started

## Prerequisites

1. Clone the repo

```shell
git clone https://github.com/star145s/ai_factory.git
```

2. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

3. Install the requirements. From your virtual environment, run

```shell
cd ai_factory
python -m pip install -e .
```

Note: flash-attn may not have their dependencies set up correctly. If you run into issues try installing those requirements separately first:

```shell
pip install packaging
pip install wheel
pip install torch
```

4. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

**Step 3: Create a `.env` file in the `ai_factory` Directory**

1. Navigate to your `ai_factory` directory where you want to save the environment file.
2. Create a new file named `.env` in this directory (if it doesn’t already exist). You can do this from the command line using:

   ```bash
   touch .env
   ```

3. Open the `.env` file with your preferred text editor and add the following lines:

    ```bash
    HF_TOKEN="<YOUR_HF_TOKEN_HERE>"
    ```

4. Save and close the file.

This `.env` file now securely holds your access tokens, allowing scripts in the `factory` directory to load it automatically if they’re set up to read environment variables.

## Running the Validator

### With auto-updates

We highly recommend running the validator with auto-updates. This will help ensure your validator is always running the latest release, helping to maintain a high vtrust.

Prerequisites:

1. To run with auto-update, you will need to have [pm2](https://pm2.keymetrics.io/) installed.
2. Make sure your virtual environment is activated. This is important because the auto-updater will automatically update the package dependencies with pip.
3. Make sure you're using the main branch: `git checkout main`.

From the ai_factory folder:

```shell
pm2 start --name net-80vali-updater --interpreter python start_validator.py -- --pm2_name net80-vali --wallet.name coldkey --wallet.hotkey hotkey 
```

This will start a process called `net80-vali-updater`. This process periodically checks for a new git commit on the current branch. When one is found, it performs a `pip install` for the latest packages, and restarts the validator process (who's name is given by the `--pm2_name` flag)

### Without auto-updates

If you'd prefer to manage your own validator updates...

From the ai_factory folder:

```shell
pm2 start python -- ./neurons/validator.py --wallet.name coldkey --wallet.hotkey hotkey
```
