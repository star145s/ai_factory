import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from taoverse.model.competition.data import Competition, ModelConstraints
from taoverse.model.competition.epsilon import LinearDecay
from taoverse.model.eval.normalization import NormalizationId
from taoverse.model.eval.task import EvalTask
from transformers import (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

import factory as fact
from competitions.data import CompetitionId
from factory.datasets.ids import DatasetId
from factory.eval.method import EvalMethodId

# ---------------------------------
# Project Constants.
# ---------------------------------

# Release
__version__ = "0.0.0"

# Validator schema version
__validator_version__ = "0.0.0"
version_split = __validator_version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The validator WANDB project.
WANDB_PROJECT = "ai-factory-validators"

# The uid for this subnet.
SUBNET_UID = 80

# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent

# Minimum stake to consider a validator when checking for miners with weights.
WEIGHT_SYNC_VALI_MIN_STAKE = 100_000

# Minimum percent of weight on a vali for a miner to be considered a top miner.
# Since there can be multiple competitions at different reward percentages we can't just check biggest.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.05

# Validator eval batch size.
BATCH_SIZE = 1
MAX_BATCHES_PER_DATASET = 2
# Validators number of pages to eval over miners on each step.

# These well be used after activation block
PAGES_PER_EVAL_RES = 1

# A mapping of block numbers to the supported model types as of that block.
ALLOWED_MODEL_TYPES_1 = {
    LlamaForCausalLM,
    Qwen2ForCausalLM,
}

# Emission distribution
TRAINING_WEIGHT = 0
CONFERENCE_WEIGHT = 0.2
WORKSHOP_WEIGHT = 0.8

# Synchronize on blocks roughly every 30 minutes.
SYNC_BLOCK_CADENCE = 150
# Delay at least as long as the sync block cadence with an additional buffer.
EVAL_BLOCK_DELAY = SYNC_BLOCK_CADENCE #+ 100

MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.RES3B_MODEL: ModelConstraints(
        max_model_parameter_size=4_000_000_000,
        min_model_parameter_size=3_000_000_000,
        sequence_length=10,
        allowed_architectures=ALLOWED_MODEL_TYPES_1,
        tokenizer="ai-factory/giant",
        kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu"
            #"attn_implementation": "flash_attention_2",
        },
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0005, 72000),
        max_bytes=29 * 1024 * 1024 * 1024,
    ),
}

# Schedule of competitions by block.
COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                CompetitionId.RES3B_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.RES3B_MODEL],
                1,
                eval_tasks=[
                    EvalTask(
                        name="ARXIV",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.ARXIV,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_pages": PAGES_PER_EVAL_RES,
                        },
                        weight=0.4,
                    ),
                    EvalTask(
                        name="REASONING",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.REASONING,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_pages": PAGES_PER_EVAL_RES,
                        },
                        weight=0.3,
                    ),
                    EvalTask(
                        name="STACKEXCHANGE",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.STACKEXCHANGE,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_pages": PAGES_PER_EVAL_RES,
                        },
                        weight=0.3,
                    ),
                ],
            ),
        ],
    ),
]

for block_and_competitions in COMPETITION_SCHEDULE_BY_BLOCK:
    assert math.isclose(
        sum(competition.reward_percentage for competition in block_and_competitions[1]),
        1.0,
    )
    for comp in block_and_competitions[1]:
        assert math.isclose(
            sum(task.weight for task in comp.eval_tasks),
            1.0,
        )


# The number of run steps to log to single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# validator weight moving average term
alpha = 0.5
# validator scoring exponential temperature
# 0.01 gives ~96% to best model with only ~3 receiving any weights.
temperature = 0.01
# validator eval batch min to keep for next loop.
sample_min = 0
# Max number of uids that can be either pending eval or currently being evaluated.
# We allow the sample_min per competition + 10 additional models to be held at any one time.
updated_models_limit = 0 #sample_min * len(MODEL_CONSTRAINTS_BY_COMPETITION_ID) + 10
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# Number of blocks required between retrying evaluation of a model.
model_retry_cadence = 300  # Roughly 1 hour
# How frequently to check the models given weights by other large validators.
scan_top_model_cadence = dt.timedelta(minutes=30)
# Min Different Score
min_diff = 0.001
# Max Different Score from top model
max_diff = 0.05
