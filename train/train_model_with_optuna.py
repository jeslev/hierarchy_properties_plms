import os

import optuna
from optuna.trial import TrialState

from train.model import Model
import argparse
from sentence_transformers import InputExample
from datasets import Dataset

import pandas as pd

FINAL_ACC = -1
PARAMS = None

def get_dataset(datapath, property=-1):
    
    def add_prefix(example):
        example["sentence1"] = example['ent_1']+" is defined as "+ example["def_1"]
        example["sentence2"] = example['ent_2']+" is defined as "+ example["def_2"]
        example["sentence3"] = example['ent_3']+" is defined as "+ example["def_3"]
        return example
    
    trainfile = os.path.join(datapath,'train_hard_soft_neg_sample_5x_p123689.json')
    devfile = os.path.join(datapath,'dev_hard_neg.json')
    #testfile = os.path.join(datapath,'test_hard_neg.json')
    
    df = pd.read_json(trainfile, orient='index')
    df = df[df['valid'] >= 0] # only positives and soft positives
    df_dev = pd.read_json(devfile, orient='index')

    if property != -1:
        df = df[df['prop'] == property]
        df_dev = df_dev[df_dev['prop'] == property]

    train_dataset = Dataset.from_pandas(df)
    dev_dataset = Dataset.from_pandas(df_dev)
    #df_test = pd.read_json(testfile, orient='index')
    #test_dataset = Dataset.from_pandas(df_test)
    
    # Adapt phrases
    ftrain_dataset = train_dataset.map(add_prefix)
    fdev_dataset = dev_dataset.map(add_prefix)
    #ftest_dataset = test_dataset.map(add_prefix)
    
    # Create inputs
    train_samples, dev_samples, test_samples = [], [], []
    for row in ftrain_dataset:
        inputrow = [ row['sentence1'],row['sentence2'],row['sentence3'] ]
        train_samples.append(InputExample(texts=inputrow))
    for row in fdev_dataset:
        inputrow = [ row['sentence1'],row['sentence2'],row['sentence3'] ]
        dev_samples.append(InputExample(texts=inputrow))
    # for row in ftest_dataset:
    #     inputrow = [ row['sentence1'],row['sentence2'],row['sentence3'] ]
    #     test_samples.append(InputExample(texts=inputrow))
    
    return train_samples, dev_samples#, test_samples
    

def get_dataset_taxonomy(datapath):    
    devfile = os.path.join(datapath,'bansal_with_defs_dev.json')
    testfile = os.path.join(datapath,'bansal_with_defs_test.json')
    
    all_elements = []
    for ifile in [devfile,testfile]:
        df = pd.read_json(ifile, orient='index')
        tree_ids = set(df["treeid"])
        elements = {}
        for tree_id in tree_ids:
            subtree = df[df['treeid']==tree_id] # get subtree
            #get words + definitions
            map_elements = {}
            golden_edges = []
            for idx,row in subtree.iterrows():
                phrase1 = row['father'] +" is defined as "+row['father_definition']
                phrase2 = row['child'] +" is defined as "+row['child_definition']
                map_elements[row['father']] = phrase1
                map_elements[row['child']] = phrase2
                golden_edges.append((row['father'],row['child']))
            
            elements[tree_id] = [golden_edges, map_elements]
        all_elements.append(elements)
    return all_elements[0],all_elements[1]
            

def objective(trial):
    params = PARAMS
    print(params)
    # Generate the optimizers.
    optimizer_name = "AdamW"#trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    lr = trial.suggest_float("lr", 1e-7, 0.1, log=True)
    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size=trial.suggest_categorical('batch',[32,64])#,32,64,128])
    epochs = trial.suggest_int("epochs",2,5, log=True)
    warmup_ratio = 0.1#trial.suggest_categorical('warmup',[0.05,0.1])
    
    
    # Get the dataset.
    print("Obtaning dataset")
    #train_samples, dev_samples, test_samples = get_dataset(params.dataset_base)
    train_samples, dev_samples = get_dataset(params.dataset_base)
    dev_taxonomy, test_taxonomy = get_dataset_taxonomy(params.dataset_base) # Elements with {[treeid]: golden edges list, list of definitions}

    # Generate the model.
    print("Preparing model")
    gen_model = Model(max_len=params.max_len, lower_case=params.lower_case, batch_size=batch_size, 
                      epochs=epochs, evaluation_steps=6000, learning_rate=lr)
    #gen_model.create_model_sentencetransformer(model_str=params.model_str,pooling_mode='mean', optimizer=optimizer_name)
    gen_model.create_model_sentencetransformer(model_str=params.model_str,pooling_mode='cls', optimizer=optimizer_name)
    
    # Configure the training
    train_loader = gen_model.get_dataloader(train_samples)

    if params.evaluator == 'taxonomy':
        gen_model.set_taxonomy_evaluator(dev_taxonomy)
    elif params.evaluator == 'triplet':
        gen_model.set_triplet_evaluator(dev_samples)
    elif params.evaluator == 'triplet_avg':
        gen_model.set_triplet_evaluator_hf(dev_samples)
        
    # 88476 samples, by batch 16 = 5520.75 batches (steps)   1 epoch
    #output_dir = './output_models'
    
    def get_scores(accuracy, epoch, steps):
        if steps==-1:
            trial.report(accuracy, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        global FINAL_ACC
        FINAL_ACC = accuracy
        print(accuracy,epoch,steps)
        #return score, epoch, steps
    
    print("Launch training")
    gen_model.train(train_loader, get_scores, warmup_ratio)
    print("Training finished. Final accuracy", FINAL_ACC)

    return FINAL_ACC


def parse_args():
    p = argparse.ArgumentParser(description='Train hierarchy properties', add_help=False)
    p.add_argument('--study_name', type=str, help='Name of the Optuna study.', default="hierarchy-taxo")
    p.add_argument('--n_trials', type=int, help='Number of trials.', default=50)
    p.add_argument('--db_name', type=str, help='Name of the Optuna database.', default="example.db")
    p.add_argument('--max_len', type=int, help='Max sequence length.', default=256)
    p.add_argument('--lower_case', type=bool, help='Lowercase tokenizer?', default=False)
    p.add_argument('--model_str', type=str, help='Model path', default='')
    p.add_argument('--evaluator', type=str, help='Type of evaluator to use. It can be [taxonomy, triplet, triplet_avg', default='taxonomy')
    p.add_argument('--dataset_base', type=str, help='Path to root data folder.', default='/users/cost/jlovonme/data/hierarchy_props')
    return p.parse_args()


if __name__ == "__main__":
    
    PARAMS = parse_args()
    study = optuna.create_study(study_name=PARAMS.study_name,
                                storage=f'sqlite:///{PARAMS.db_name}',
                                load_if_exists=True,
                                direction="maximize")
    study.optimize(objective, n_trials=PARAMS.n_trials,timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
