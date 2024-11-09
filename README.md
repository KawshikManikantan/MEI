## General Information:
This repository consists of the code-base for the trained models that perform [Major Entity Identification](https://arxiv.org/abs/2406.14654). The code-base for few-shot LLM inference will be released soon.

The repository consists of two major components:
- Coreference-based ([longdoc](https://github.com/shtoshni/fast-coref)) baselines with inference-time mapping to the entities.
- MEIRa models with direct mappings to entities.

We use hydra configs across the project. Coreference and MEIRa models have their training configs structured with Hydra.
## Repo Details
The repo consists of the following folders:
- `mei-data`: Contains raw_data of datasets, present in this drive [link]()
- `longdoc`: Contains code for training a longdoc coreference model
- `MEIRa`: Contains code for training a MEIRa model
- `src`: Contains code that utilizes models trained above to generate MEI outputs
- `models`: Contains coreference (coref_) and MEI (mei_) models
  

## Models
Clone the models from their huggingface repositories:
`MEIRa-S`: [https://huggingface.co/KawshikManikantan/meira-s](https://huggingface.co/KawshikManikantan/meira-s)
`MEIRa-H`: [https://huggingface.co/KawshikManikantan/meira-h](https://huggingface.co/KawshikManikantan/meira-h)

Create a directory `models` within the base directory and place the models within it.

## Data
Download the dataset from [gdrive](https://drive.google.com/drive/folders/1vaVwHhMaDDXLw0rkLzTm5-AOSBDJVNJF?usp=sharing) 

## How to train models?
Training code is quite similar to [longdoc](https://github.com/shtoshni/fast-coref). Check the README for more details. \
We have added a few keys for more convenience:
- `device`: specifies the devices to train the model on -- defaults to `cuda:0`
- `key`: In addition to unique model identifiers, add a key to uniquely label the experiment -- defaults to null str 

`Note: {} -- denotes a specific instance of a particular sub-config repository.`

### longdoc

**Template:**

```
cd longdoc/
python main.py experiment={experiment_name} model/doc_encoder/transformer={backbone_name} use_wandb={True/False} device={cpu/cuda:i/auto} key={label}
```

**Example:**
```
cd longdoc/
python main.py experiment=ontonotes_pseudo model/doc_encoder/transformer=longformer_large use_wandb=True devie="cuda:1" key="onto_train"
```

### MEIRa

**Template:**

```
cd MEIRa/
python main.py experiment={experiment_name} model/doc_encoder/transformer={backbone_name} use_wandb={True/False} device={cpu/cuda:i/auto} key={label} model.memory.type={hybrid/static}
```

**Example:**
```
cd MEIRa/
python main.py experiment=ontonotes_pseudo model/doc_encoder/transformer=longformer_large use_wandb=True devie="cuda:1" key="onto_train" model.memory.type=static
```

## How to infer?

**Get coreference clusters:**
```
cd longdoc/
python main.py experiment={experiment_eval} model/doc_encoder/transformer={model_name} use_wandb=False train=False device={cpu/cuda:i/auto} paths.model_dir={model_dir} model.memory.type=static
```