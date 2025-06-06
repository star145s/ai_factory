import asyncio
import functools
import os
from typing import Optional

import bittensor as bt

import taoverse.utilities.logging as logging
from taoverse.model.data import ModelId, ModelMetadata
from taoverse.model.storage.model_metadata_store import ModelMetadataStore
from taoverse.utilities import utils


class ChainModelMetadataStore(ModelMetadataStore):
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        subnet_uid: int,
        # Wallet is only needed to write to the chain, not to read.
        wallet: Optional[bt.wallet] = None,
    ):
        self.subtensor = subtensor
        self.wallet = wallet
        self.subnet_uid = subnet_uid

    async def store_model_metadata(
        self,
        hotkey: str,
        model_id: ModelId,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        ttl: int = 60,
    ):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        data = model_id.to_compressed_str()

        commit_partial = functools.partial(
            self.subtensor.commit,
            self.wallet,
            self.subnet_uid,
            data,
        )


        utils.run_in_thread(commit_partial, ttl)

    async def retrieve_model_metadata(
        self, uid: int, hotkey: str, ttl: int = 60
    ) -> Optional[ModelMetadata]:
        """Retrieves model metadata on this subnet for specific hotkey"""

        # Wrap calls to the subtensor in a thread with a timeout to handle potential hangs.
        metadata_partial = functools.partial(
            bt.core.extrinsics.serving.get_metadata,
            self.subtensor,
            self.subnet_uid,
            hotkey,
        )

        commitment_partial = functools.partial(
            self.subtensor.get_commitment,
            self.subnet_uid,
            uid,
        )

        metadata = utils.run_in_thread(metadata_partial, ttl)

        if not metadata:
            return None

        chain_str = utils.run_in_thread(commitment_partial, ttl)

        model_id = None

        try:
            model_id = ModelId.from_compressed_str(chain_str)
        except:
            # If the metadata format is not correct on the chain then we return None.
            logging.trace(
                f"Failed to parse the metadata on the chain for hotkey {hotkey}."
            )
            return None

        model_metadata = ModelMetadata(id=model_id, block=metadata["block"])

        return model_metadata


# Can only commit data every ~20 minutes.
async def test_store_model_metadata():
    """Verifies that the ChainModelMetadataStore can store data on the chain."""
    model_id = ModelId(
        namespace="TestPath",
        name="TestModel",
        competition_id=1,
        hash="TestHash1",
        commit="1.0",
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=net_uid,
        wallet=wallet,
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(
        hotkey=hotkey,
        model_id=model_id,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    print(f"Finished storing {model_id} on the chain.")


async def test_retrieve_model_metadata():
    """Verifies that the ChainModelMetadataStore can retrieve data from the chain."""
    expected_model_id = ModelId(
        namespace="TestPath",
        name="TestModel",
        competition_id=1,
        hash="TestHash1",
        commit="1.0",
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured hotkey/uid for the test.
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    hotkey_address = os.getenv("TEST_HOTKEY_ADDRESS")
    model_uid = int(os.getenv("TEST_MODEL_UID"))

    # Do not require a wallet for retrieving data.
    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=net_uid,
        wallet=None,
    )

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(model_uid, hotkey_address)

    print(f"Expecting matching model id: {expected_model_id == model_metadata.id}")


# Can only commit data every ~20 minutes.
async def test_roundtrip_model_metadata():
    """Verifies that the ChainModelMetadataStore can roundtrip data on the chain."""
    model_id = ModelId(
        namespace="TestPath",
        name="TestModel",
        competition_id=1,
        hash="TestHash1",
        commit="1.0",
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))
    model_uid = int(os.getenv("TEST_MODEL_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=net_uid,
        wallet=wallet,
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(
        hotkey=hotkey,
        model_id=model_id,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(model_uid, hotkey)

    print(f"Expecting matching metadata: {model_id == model_metadata.id}")


if __name__ == "__main__":
    # Can only commit data every ~20 minutes.
    asyncio.run(test_roundtrip_model_metadata())
    asyncio.run(test_store_model_metadata())
    asyncio.run(test_retrieve_model_metadata())
