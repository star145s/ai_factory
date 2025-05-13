from typing import Any, Dict

from factory.dataset import (
    SubsetLoader,
    SubsetArxivLoader,
    SubsetReasoningLoader,
    SubsetStackExchangeLoader
)

from factory.datasets.ids import DatasetId
from transformers import PreTrainedTokenizerBase


class DatasetLoaderFactory:
    @staticmethod
    def get_loader(
        dataset_id: DatasetId,
        dataset_kwargs: Dict[str, Any],
        seed: int,
        sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> SubsetLoader:
        """Loads data samples from the appropriate dataset."""

        match dataset_id:
            case DatasetId.ARXIV:
                return SubsetArxivLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.REASONING:
                return SubsetReasoningLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case DatasetId.STACKEXCHANGE:
                return SubsetStackExchangeLoader(
                    random_seed=seed,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    **dataset_kwargs,
                )
            case _:
                raise ValueError(f"Unknown dataset_id: {dataset_id}")
