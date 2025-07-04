import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

import taoverse.utilities.logging as logging
from taoverse.model.data import Model, ModelId
from taoverse.model.storage.disk import utils
from taoverse.model.storage.local_model_store import LocalModelStore

class DiskModelStore(LocalModelStore):
    """Local storage based implementation for storing and retrieving a model on disk."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(utils.get_local_miners_dir(base_dir), exist_ok=True)

    def get_path(self, hotkey: str) -> str:
        """Returns the path to where this store would locate this hotkey."""
        return utils.get_local_miner_dir(self.base_dir, hotkey)

    def store_model(self, hotkey: str, model: Model) -> ModelId:
        """Stores a trained model locally."""

        save_directory = utils.get_local_model_snapshot_dir(
            self.base_dir, hotkey, model.id
        )

        # Note that the revision argument here does not affect the directory path like with hugging face downloads.
        model.pt_model.save_pretrained(
            save_directory=save_directory,
            revision=model.id.commit,
            safe_serialization=True,
        )

        # If the model has a tokenizer also save that.
        if model.tokenizer is not None:
            model.tokenizer.save_pretrained(
                save_directory=save_directory,
                revision=model.id.commit,
                safe_serialization=True,
            )

        # Return the same model id used as we do not edit the commit information.
        return model.id

    def retrieve_model(
        self, hotkey: str, model_id: ModelId, kwargs: Dict[str, Any]
    ) -> Model:
        """Retrieves a trained model locally."""

        pretrained_model_name_or_path = utils.get_local_model_snapshot_dir(
            self.base_dir, hotkey, model_id
        )
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    revision=model_id.commit,
                    local_files_only=True,
                    use_safetensors=True,
                    **kwargs
                )
        except Exception as e:
            if os.path.exists(pretrained_model_name_or_path):
                shutil.rmtree(pretrained_model_name_or_path)
            raise e

        # Always try to retrieve a tokenizer from the model directory. If we do not find one leave it None on the Model.
        tokenizer = None
        try:
            # Do not use the kwargs for the model load here. If needed in the future a separate kwargs can be plumbed.
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                revision=model_id.commit,
                local_files_only=True,
                use_safetensors=True,
            )
        except Exception:
            pass

        return Model(id=model_id, pt_model=model, tokenizer=tokenizer)

    def delete_unreferenced_models(
        self, valid_models_by_hotkey: Dict[str, ModelId], grace_period_seconds: int
    ):
        """Check across all of local storage and delete unreferenced models out of grace period."""
        # Expected directory structure is as follows.
        # self.base_dir/models/hotkey/models--namespace--name/snapshots/commit/config.json + other files.

        # Create a set of valid model paths up to where we expect to see the actual files.
        valid_model_paths = set()
        for hotkey, model_id in valid_models_by_hotkey.items():
            valid_model_paths.add(
                utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model_id)
            )

        # For each hotkey path on disk using listdir to go one level deep.
        miners_dir = Path(utils.get_local_miners_dir(self.base_dir))
        hotkey_subfolder_names = [d.name for d in miners_dir.iterdir() if d.is_dir()]

        for hotkey in hotkey_subfolder_names:
            try:
                # Reconstruct the path from the hotkey
                hotkey_path = utils.get_local_miner_dir(self.base_dir, hotkey)

                # If it is not in valid_hotkeys and out of grace period remove it.
                if hotkey not in valid_models_by_hotkey:
                    deleted_hotkey = utils.remove_dir_out_of_grace(
                        hotkey_path, grace_period_seconds
                    )
                    if deleted_hotkey:
                        logging.trace(
                            f"Removed directory for unreferenced hotkey: {hotkey}."
                        )

                else:
                    # Check all the models--namespace--name subfolder paths.
                    hotkey_dir = Path(hotkey_path)
                    model_subfolder_paths = [
                        str(d) for d in hotkey_dir.iterdir() if d.is_dir()
                    ]

                    # Check all the snapshots subfolder paths
                    for model_path in model_subfolder_paths:
                        model_dir = Path(model_path)
                        snapshot_subfolder_paths = [
                            str(d) for d in model_dir.iterdir() if d.is_dir()
                        ]

                        # Check all the commit paths.
                        for snapshot_path in snapshot_subfolder_paths:
                            snapshot_dir = Path(snapshot_path)
                            commit_subfolder_paths = [
                                str(d) for d in snapshot_dir.iterdir() if d.is_dir()
                            ]

                            # Reached the end. Check all the actual commit subfolders for the files.
                            for commit_path in commit_subfolder_paths:
                                if commit_path not in valid_model_paths:
                                    deleted_model = utils.remove_dir_out_of_grace(
                                        commit_path, grace_period_seconds
                                    )
                                    if deleted_model:
                                        logging.trace(
                                            f"Removing directory for unreferenced model at: {commit_path}."
                                        )
            except Exception:
                # Catch the exception so we continue with the rest of the cleanup.
                logging.warning(traceback.format_exc())
