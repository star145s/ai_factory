import os
import random
import time
import typing


import bittensor as bt
import boto3
import numpy as np
import requests
import smart_open

from io import BytesIO
from dotenv import load_dotenv

import torch

from taoverse.utilities import logging
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from factory.datasets.fetch_arxiv import split_content 

load_dotenv()


class SubsetLoader(IterableDataset):
    """Base class for data-specific subset loader classes."""

    name: str = None  # Dataset name
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"
    max_pages: int = None

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = True,
        random_seed: typing.Optional[int] = None,
        config: str = "default",
        split: str = "train",
        requires_auth: bool = False,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples
        self.config = config
        self.split = split
        self.requires_auth = requires_auth

        # Initialize with seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        self.num_rows_per_page = 50
        self.duplicate_page_threshold = 100
        self.retry_limit = 15
        self.retry_delay = 5

        # Buffers
        self.buffer = []
        self.used_buffer = []
        self.padded_buffer = []

        # Get HF token if needed
        self.hf_token = None
        if self.requires_auth:
            self.hf_token = os.getenv("HF_TOKEN")
            if not self.hf_token:
                raise ValueError("HF_TOKEN environment variable not found")

        # Initialize request params
        self.params = self._get_default_params()

        # Fetch pages if specified
        # If the fetched pages are empty, try again until
        # we hit the retry limit.
        fetch_attempt = 1

        if self.num_pages:
            while fetch_attempt < self.retry_limit:
                self._initialize_pages()
                fetch_attempt += 1

                # Exit if the buffer has at least one batch
                if len(self.buffer) >= self.sequence_length:
                    break

                logging.warning(
                    f"All fetched pages seem to be empty or have an extremely low token count. "
                    f"Trying to fetch a new set of pages... (attempt {fetch_attempt}/{self.retry_limit})"
                )

            # If we exhaust all attempts and still don't have enough data, raise an error
            if len(self.buffer) < self.sequence_length:
                raise ValueError(
                    "Maximum retry limit for fetching pages reached. "
                    "All fetched pages seem to be empty or have an extremely low token count."
                )

    def _get_default_params(self):
        """Get default request parameters. Override if needed."""
        return {
            "dataset": self.name,
            "config": self.config,
            "split": self.split,
        }

    def _get_request_headers(self):
        """Get request headers. Override if needed."""
        headers = {}
        if self.requires_auth:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    def _initialize_pages(self):
        """Initialize pages based on loader type"""
        pages = self._sample_pages()
        self.fetch_data_for_pages(pages)

    def fetch_data_for_pages(self, pages):
        """Set the pages and fetch their data to the buffer."""
        self.pages = pages
        self.buffer = []
        for page in self.pages:
            self._fetch_data_for_page(page)

    def _fetch_data_for_page(self, page):
        """Fetch data for a single page"""
        # Handle different page types (tuple vs int)
        if isinstance(page, tuple):
            config_name, page_num, split = page
            self.params.update(
                {
                    "config": config_name,
                    "split": split,
                    "offset": page_num,
                }
            )
        else:
            self.params["offset"] = page

        self.params["length"] = self.num_rows_per_page

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(
                    self.rows_base_url,
                    params=self.params,
                    headers=self._get_request_headers(),
                    timeout=15,
                )
                response.raise_for_status()
                for row in response.json()["rows"]:
                    content = self._get_content_from_row(row)
                    contents = split_content(content, self.tokenizer)
                    
                    for content in contents:
                        input_ids = self.tokenizer(content, truncation=True)["input_ids"]
                        self.buffer += input_ids
                        self.buffer += [self.tokenizer.eos_token_id]

                break

            except requests.exceptions.RequestException as e:
                attempt += 1
                logging.warning(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _get_content_from_row(self, row):
        """Extract content from row based on dataset format. Override if needed."""
        return row["row"].get("text", row["row"].get("content"))

    def _sample_pages(self):
        """Sample random pages. Override for custom sampling logic."""
        return [random.randint(1, self.max_pages) for _ in range(self.num_pages)]

    def get_page_names(self):
        """Get page names in consistent format"""
        if not hasattr(self, "pages"):
            return []

        if isinstance(self.pages[0], tuple):
            return [
                f"{cfg_name}_{num_rows}_{split}"
                for cfg_name, num_rows, split in self.pages
            ]
        return self.pages

    def _get_pad_size(self, input_ids):
        """Get padding size for input tokens."""
        if self.pack_samples:
            return 1

        sample_size = len(input_ids)
        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        pad_size = pad_size % self.sequence_length
        return pad_size

    def _refill_padded_buffer(self):
        """Refill the padded buffer from the main buffer."""
        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]
            self.used_buffer += input_ids
            self.padded_buffer += input_ids[:-1]
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(
                input_ids=input_ids[:-1]
            )

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []
        self._refill_padded_buffer()
        return self

    def __next__(self):
        batch = []
        while len(self.padded_buffer) >= self.sequence_length:
            batch.append(self.padded_buffer[: self.sequence_length])
            self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._refill_padded_buffer()
            if len(batch) == self.batch_size:
                return np.stack(batch)
        raise StopIteration

class SubsetArxivLoader(SubsetLoader):
    max_pages: int = 15555
    name: str = "ai-factory/red_pajama_subset_arxiv_subset"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SubsetReasoningLoader(SubsetLoader):
    max_pages: int = 221994
    name: str = "ai-factory/glaiveai-reasoning-v1-20m-chat"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SubsetStackExchangeLoader(SubsetLoader):
    max_pages: int = 252808
    name: str = "ai-factory/red_pajama_subset_stackexchange_subset"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)