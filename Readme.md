# Hierarchy Properties Evaluation for PLMs

This repository contains code and models for the paper TBA


## Resources


- [Datasets](https://drive.google.com/drive/folders/1FSqFbjfG6GkK3tQ876Ylg-duUO5vIkFN?usp=sharing) **properties** folder.
- [Trained Models with hierarchies](https://drive.google.com/drive/folders/1rflS3vdWbrSpDO3tLeiFa-yaIM7BbzVB?usp=sharing)
- Other (representations, individual scores, etc) [link](https://drive.google.com/drive/folders/1b08QHuOOWe97C7Ny4k_lbBAWZmTZtfE2?usp=sharing)

-----

The following contains command examples to reproduce different parts of our pipeline.


## Generate data (data_generator folder)
It contains scripts and notebook to create the data. We provide also the generated dataset for our work and the input files needed for the generation.
It uses as base 2 files: bansal14_trees.csv and definitions_bansal.json

## Evaluation:
    1. Extract Representation (improve code to define exact representations - type and layers)

Offline extraction of PLMs representations. First run
> mkdir -p output/model_representations
> python run_extract_representations.py

This will save representations in model_vectors folder.


Once the vectors are extracted, we compute the properties accuracy with:
> mkdir -p output/props_scores
> python run_evaluate_props.py

Define in the file .py which configurations to run.


See evaluation_analysis notebook for results and qualitative analysis.


## Train (train folder)

### Inject triplet properties by fine-tuning using sentence transformers  
To train models with optuna
> python train_model_with_optuna.py --model_str model_name

To train models individually (and save them)


> python -m train.train_model --pool cls --model_str bert-base-cased --lr 2e-7 --bs 32 --epochs 1 --output_dir out --dataset_base data_generator/properties --property -1




### Extract
Extract the trained sentence transformer as transformer

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name)
# Extract HugginFace base model
hf_model = model[0].auto_model
hf_model.save_pretrained(path)
```

## Fine-tune downstream tasks

### QA tasks
Code for RACE and Probes MCQA from this [repo]([repo](https://github.com/allenai/semantic_fragments)https://github.com/allenai/semantic_fragments).
