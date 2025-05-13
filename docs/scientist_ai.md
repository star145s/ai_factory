# Research Track: Developing AI for Original Scientific Idea Generation

The **Research Track** aims to develop artificial intelligence (AI) systems capable of generating original, feasible, and highly novel scientific ideas. Traditional large language models (LLMs) often face criticism for their tendency to memorize information and produce hallucinations, limiting their ability to reason and innovate like humans. To address these challenges, we are exploring advanced training methodologies, such as those employed in models like DeepSeek-R1, to enhance reasoning capabilities through reinforcement learning (RL).

DeepSeek-R1 has demonstrated that applying RL directly to a base model can lead to the emergence of sophisticated reasoning behaviors, including self-verification, reflection, and extended chain-of-thought generation. This approach has enabled the model to achieve performance levels comparable to leading AI systems, such as OpenAI's o1–1217, while reducing reliance on supervised fine-tuning. [Source](https://arxiv.org/abs/2501.12948)

## Initial Phase (Completed)

In the initial phase of the Research Track, participants will pretrain a 500-million-parameter model using a subset of the arXiv dataset from togethercomputer/RedPajama-Data-1T . This foundational step will establish a robust base for subsequent training phases.

## Second Phase (Ongoing)

In this phase, the model scale will be increased to 4 billion parameters, and the training datasets will be extended. Below is the competition information:

### Model Constraints:
- **Model types**: Qwen, LLaMA, DeepSeek  
- **Model size**: 3–4 billion parameters  
- **Tokenizer**: `ai-factory/giant`

### Training and Testing Data:
- `ai-factory/red_pajama_subset_arxiv_subset`: ArXiv papers truncated to 4096-token chunks.  
- `ai-factory/red_pajama_subset_stackexchange_subset`: Samples are formatted using a chat template with two roles: `"user"` (representing questions) and `"other"` (representing answers).  
- `ai-factory/glaiveai-reasoning-v1-20m-chat`: Samples are formatted using a chat template with two roles: `"user"` (representing questions) and `"me"` (representing the AI).  


## Future Plans

- **Developing Novelty and Correctness Metrics:** Establishing new metrics to assess the originality and accuracy of the generated scientific ideas.

By integrating these strategies, the Research Track aspires to push the boundaries of AI's capability to autonomously develop innovative scientific concepts, thereby contributing significantly to the advancement of artificial intelligence in scientific research.


# Miner Instruction

Miners train locally and periodically publish their best model to hugging face and commit the metadata for that model to the Bittensor chain.

Miners can only have one model associated with them on the chain for evaluation by validators at a time. The list of allowed model types by block can be found in [constants/__init__.py](). Other relevant constraints are also listed in that file.

The communication between a miner and a validator happens asynchronously chain and therefore Miners do not need to be running continuously. Validators will use whichever metadata was most recently published by the miner to know which model to download from hugging face.

# System Requirements

Miners will need enough disk space to store their model as they work on. Max size of model is defined in [constants/__init__.py]

# Getting started

## Prerequisites

1. Get a Hugging Face Account: 

Miner and validators use hugging face in order to share model state information. Miners will be uploading to hugging face and therefore must attain a account from [hugging face](https://huggingface.co/) along with a user access token which can be found by following the instructions [here](https://huggingface.co/docs/hub/security-tokens).

Make sure that any repo you create for uploading is public so that the validators can download from it for evaluation.

2. Clone the repo

```shell
git clone https://github.com/star145s/ai_factory.git
```

3. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

4. Install the requirements. From your virtual environment, run
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

5. Make sure you've [created a Wallet](https://docs.bittensor.com/working-with-keys) and [registered a hotkey](https://docs.bittensor.com/miners/).

---

# Running the Miner

The mining script uploads a model to hugging face which will be evaluated by validators.

See [Validator Psuedocode](docs/validator.md#validator) for more information on how they the evaluation occurs.

## Env File

The Miner requires a .env file with your hugging face access token in order to upload models.

Create a `.env` file in the `pretraining` directory and add the following to it:
```shell
HF_ACCESS_TOKEN="YOUR_HF_ACCESS_TOKEN"
```

## Starting the Miner

You can manually upload with the following command:
```shell
python upload_model.py --load_model_dir <path to model> --competition_id 0 --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey
```

Note: We recommend keeping your hugging face repo private until after you have committed your metadata to the chain. This ensures other miners are unable to upload your model as their own until a later block. Adding the `--update_repo_visibility` flag will also automatically attempt to update the hugging face repo visibility to public after committing to the chain.

Note: If you are not sure about the competition ID, you can add the `--list_competitions` flag to get a list of all competitions. You can also check out competition IDs in [competitions/data.py].

---

## Running a custom Miner

The list of allowed model types by block can be found in [constants/__init__.py]

In that file are also the constraints per block for
1. The details of every competition.
2. All the model constraints (sequence length, allowed model types, dataset, etc)

The `factory/mining.py` file has several methods that you may find useful. Example below.

```python
import factory as fact
import bittensor as bt
from transformers import PreTrainedModel

# Load a model from another miner.
model: PreTrainedModel = await pt.mining.load_remote_model(uid=123, download_dir="mydir")

# Save the model to local file.
fact.mining.save(model, "model-foo/")

# Load the model from disk.
fact.mining.load_local_model("model-foo/", **kwargs)

# Publish the model for validator evaluation.
wallet = bt.wallet()
await fact.mining.push(model, repo="jdoe/my-repo", wallet=wallet, competition_id=1)

# Get the URL to the best model for a given competition
best_uid = fact.graph.best_uid(competition_id=1)
print(await fact.mining.get_repo(best_uid))
```