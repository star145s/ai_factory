# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Tools for performing validation over models.

import dataclasses
import math
import typing

import taoverse.utilities.logging as logging
import torch
from taoverse.model.competition.epsilon import EpsilonFunc
from taoverse.model.data import Model
from taoverse.model.eval.normalization import normalize_score
from taoverse.model.eval.task import EvalTask

from factory.eval.method import EvalMethodId, compute_text_loss
from factory.eval.sample import EvalSample

def iswin(
    loss_i: float,
    loss_j: float,
    block_i: int,
    block_j: int,
    epsilon_func: EpsilonFunc,
    current_block: int,
) -> bool:
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
        epsilon_func (EpsilonFunc): Function that determines how much advantage to give to the earlier block.
        current_block: The current block.

    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust loss based on timestamp and epsilon.
    loss_i = (
        (1 - epsilon_func.compute_epsilon(current_block, block_i)) * loss_i
        if block_i < block_j
        else loss_i
    )
    loss_j = (
        (1 - epsilon_func.compute_epsilon(current_block, block_j)) * loss_j
        if block_j < block_i
        else loss_j
    )
    return loss_i < loss_j

def is_close(loss_i, loss_j, block_i, block_j, min_diff):
    if abs(loss_i - loss_j) < min_diff and block_i > block_j:
        return False 
    return True
    
def compute_wins(
    uids: typing.List[int],
    uid_to_score: typing.Dict[int, float],
    uid_to_block: typing.Dict[int, int],
    epsilon_func: EpsilonFunc,
    current_block: int,
    min_diff: float,
    max_diff: float
) -> typing.Tuple[typing.Dict[int, int], typing.Dict[int, float]]:
    """
    Computes the number of wins and win rate for each UID based on pairwise comparisons 
    of their scores and block advantages.

    Args:
        uids (List[int]): List of unique identifiers for models.
        uid_to_score (Dict[int, float]): Mapping from UID to its score.
        uid_to_block (Dict[int, int]): Mapping from UID to its block number.
        epsilon_func (EpsilonFunc): Function to compute acceptable score difference
                                    based on block difference.
        current_block (int): The current block number.

    Returns:
        Tuple[Dict[int, int], Dict[int, float]]: 
            - Dictionary mapping each UID to its number of wins.
            - Dictionary mapping each UID to its win rate.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0.0 for uid in uids}
    min_score = min(uid_to_score.values())

    for uid_i in uids:
        # Skip models that are significantly worse than the best one
        if abs(uid_to_score[uid_i] - min_score) > max_diff:
            win_rate[uid_i] = 0.0
            continue

        # Skip models that are too close to any other model (to avoid uncertainty)
        if any(
            uid_i != uid_j and is_close(
                uid_to_score[uid_i],
                uid_to_score[uid_j],
                uid_to_block[uid_i],
                uid_to_block[uid_j],
                min_diff
            )
            for uid_j in uids
        ):
            win_rate[uid_i] = 0.0
            continue

        # Count wins against all other models
        total_matches = 0
        for uid_j in uids:
            if uid_i == uid_j:
                continue
            total_matches += 1
            if iswin(
                uid_to_score[uid_i],
                uid_to_score[uid_j],
                uid_to_block[uid_i],
                uid_to_block[uid_j],
                epsilon_func,
                current_block,
            ):
                wins[uid_i] += 1

        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 1.0

    return wins, win_rate

@dataclasses.dataclass
class ScoreDetails:
    """Details of the score for a model."""

    raw_score: typing.Optional[float] = None
    norm_score: typing.Optional[float] = None
    weighted_norm_score: typing.Optional[float] = None
    num_samples: int = 0


def score_model(
    model: Model,
    evals: typing.List[EvalTask],
    samples: typing.List[typing.List[EvalSample]],
    device: str,
) -> typing.Tuple[float, dict]:
    """Scores a model based on the provided eval tasks.

    Args:
        model (torch.nn.Module): The model to score.
        evals (list): A list of EvalTasks to score the model on.
        samples (list): A list of samples to use for scoring for the eval tasks. Must be the same length as evals.
        competition (Competition): The competition to score the model for.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        tuple: A tuple containing the score and a dictionary of score details."""

    if len(evals) != len(samples):
        raise ValueError("Number of eval tasks and samples must match.")

    if not model.tokenizer:
        raise ValueError("Model does not have a tokenizer")

    with torch.inference_mode():
        model.pt_model.to(device)
        model.pt_model.eval()

        score = 0
        score_details = {task.name: ScoreDetails() for task in evals}
        tokenizer = model.tokenizer

        for task, samples in zip(evals, samples):
            logging.trace(f"Scoring model on task: {task.name}")
            match task.method_id:
                case EvalMethodId.TEXT_LOSS:
                    raw_score = compute_text_loss(
                        model=model.pt_model,
                        batches=samples,
                        device=device,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                case _:
                    raise ValueError(f"Unhandled evaluation method {task.method_id}.")
            # Normalize score
            normalized_score = normalize_score(
                raw_score, task.normalization_id, task.normalization_kwargs
            )
            weighted_norm_score = normalized_score * task.weight

            score += weighted_norm_score
            score_details[task.name] = ScoreDetails(
                raw_score=raw_score,
                norm_score=normalized_score,
                weighted_norm_score=weighted_norm_score,
                num_samples=len(samples),
            )

    return score, score_details


def compute_competitive_uids(
    uid_to_score: typing.Dict[int, float],
    uid_to_block: typing.Dict[int, int],
    epsilon_func: EpsilonFunc,
) -> typing.List[int]:
    """
    Computes the list of any uids that may at one point be the top model.

    Parameters:
        uid_to_score (dict): A dictionary of score for each uid over all batches.
        uid_to_block (dict): A dictionary of blocks for each uid.
        epsilon_func (EpsilonFunc): Function that determines how much advantage to give to the earlier block.

    Returns:
        list: A list of uids that may at one point be the top model.
    """
    # Get fully decayed loss for every model.
    fully_decayed_epsilon = 1 - epsilon_func.compute_epsilon(
        current_block=math.inf, model_block=0
    )
    fully_decayed_scores = {
        uid: uid_to_score[uid] * fully_decayed_epsilon for uid in uid_to_block
    }

    competitive_uids = []

    # Compute mean score
    mean_decayed_score = sum(fully_decayed_scores.values())/len(fully_decayed_scores) if len(fully_decayed_scores) > 0 else float("inf")
    
    for uid, loss in uid_to_score.items():
        if loss < mean_decayed_score:
            competitive_uids.append(uid)
            
    return competitive_uids
