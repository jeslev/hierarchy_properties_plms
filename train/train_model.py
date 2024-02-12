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

def get_dataset(datapath, filterprop=-1):

    def add_prefix(example):
        example["sentence1"] = example['ent_1']+" is defined as "+ example["def_1"]
        example["sentence2"] = example['ent_2']+" is defined as "+ example["def_2"]
        example["sentence3"] = example['ent_3']+" is defined as "+ example["def_3"]
        return example

    def filter_df(_df, low, high=-1):
        if high==-1:
            return _df[_df['prop']==low]
        return _df[(_df['prop']==low) | (_df['prop']==high)]

    trainfile = os.path.join(datapath,'train_hard_soft_neg_sample_5x_p123689.json')
    devfile = os.path.join(datapath,'dev_hard_neg.json')
    testfile = os.path.join(datapath,'test_hard_neg.json')

    df = pd.read_json(trainfile, orient='index')
    #df = df[df['valid'] >= 0] # only positives and soft positives  
    df = df[df['valid'] == 1] # only positives
    df_dev = pd.read_json(devfile, orient='index')
    df_test = pd.read_json(testfile, orient='index')

    if filterprop != -1:
        if filterprop==1:
            #df, df_dev, df_test = filter_df(df,1), filter_df(df_dev,1), filter_df(df_test,1)
            df = filter_df(df,1)
        elif filterprop==2:
            #df, df_dev, df_test = filter_df(df,2), filter_df(df_dev,2), filter_df(df_test,2)
            df = filter_df(df,2)
        elif filterprop==3:
            #df, df_dev, df_test = filter_df(df,3,4), filter_df(df_dev,3,4), filter_df(df_test,3,4)
            df = filter_df(df,3,4)
        elif filterprop==4:
            #df, df_dev, df_test = filter_df(df,5), filter_df(df_dev,5), filter_df(df_test,5)
            df = filter_df(df,5)
        elif filterprop==5:
            #df, df_dev, df_test = filter_df(df,6,7), filter_df(df_dev,6,7), filter_df(df_test,6,7)
            df = filter_df(df,6,7)
        elif filterprop==6:
            #df, df_dev, df_test = filter_df(df,8,9), filter_df(df_dev,8,9), filter_df(df_test,8,9)
            df = filter_df(df,8,9)

    train_dataset = Dataset.from_pandas(df)
    # Adapt phrases
    ftrain_dataset = train_dataset.map(add_prefix)

    # Create inputs
    train_samples, dev_samples, test_samples = [], {}, {}
    for row in ftrain_dataset:
        inputrow = [ row['sentence1'],row['sentence2'],row['sentence3'] ]
        train_samples.append(InputExample(texts=inputrow))

    for prop in range(1,10):

        pdf_dev = df_dev[df_dev['prop'] == prop]
        pdf_test = df_test[df_test['prop'] == prop]

        dev_dataset = Dataset.from_pandas(pdf_dev)
        test_dataset = Dataset.from_pandas(pdf_test)
        fdev_dataset = dev_dataset.map(add_prefix)
        ftest_dataset = test_dataset.map(add_prefix)

        for row in fdev_dataset:
            cur_list = dev_samples.get('p'+str(prop),[])
            inputrow = [ row['sentence1'],row['sentence2'],row['sentence3'] ]
            cur_list.append(InputExample(texts=inputrow))
            dev_samples['p'+str(prop)] = cur_list

        for row in ftest_dataset:
            cur_list = test_samples.get('p'+str(prop),[])
            inputrow = [ row['sentence1'],row['sentence2'],row['sentence3'] ]
            cur_list.append(InputExample(texts=inputrow))
            test_samples['p'+str(prop)] = cur_list


    return train_samples, dev_samples, test_samples


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


def train_model():
    params = PARAMS

    # Generate the optimizers.
    optimizer_name = "AdamW"
    lr = params.lr
    batch_size=params.bs
    epochs = params.epochs
    warmup_ratio = 0.1
    output_dir =  params.output_dir
    filter_property=params.property

    # Get the dataset.
    print("Obtaning dataset")
    train_samples, dev_samples, test_samples = get_dataset(params.dataset_base,filterprop=filter_property)
    #dev_taxonomy, test_taxonomy = get_dataset_taxonomy(params.dataset_base) # Elements with {[treeid]: golden edges list, list of definitions}

    # Generate the model.
    print("Preparing model")
    gen_model = Model(max_len=params.max_len, lower_case=params.lower_case, batch_size=batch_size,
                      epochs=epochs, evaluation_steps=-1, learning_rate=lr, output_dir=output_dir)

    # Set mean or cls according to params.pool value
    #gen_model.create_model_sentencetransformer(model_str=params.model_str,pooling_mode='mean', optimizer=optimizer_name)
    #gen_model.create_model_sentencetransformer(model_str=params.model_str,pooling_mode='cls', optimizer=optimizer_name)
    gen_model.create_model_sentencetransformer(model_str=params.model_str,pooling_mode=params.pool, optimizer=optimizer_name)

    # Configure the training
    train_loader = gen_model.get_dataloader(train_samples)
    # Setup evaluation
    gen_model.set_triplet_filtered_evaluator(dev_samples, test_samples)
    # 88476 samples, by batch 16 = 5520.75 batches (steps)   1 epoch

    print("Launch training")
    gen_model.train_and_test(train_loader, warmup_ratio)
    print("Training finished")




def parse_args():
    p = argparse.ArgumentParser(description='Train hierarchy properties', add_help=False)
    p.add_argument('--max_len', type=int, help='Max sequence length.', default=256)
    p.add_argument('--lower_case', type=bool, help='Lowercase tokenizer?', default=False)
    p.add_argument('--pool', type=str, help='mean or cls', default='cls')
    p.add_argument('--model_str', type=str, help='Model path', default='')
    p.add_argument('--lr', type=float, help='Learning rate', default=0.0)
    p.add_argument('--bs', type=int, help='Batch size', default=8)
    p.add_argument('--epochs', type=int, help='Batch size', default=4)
    p.add_argument('--output_dir', type=str, help='directory to store model', default='')
    p.add_argument('--dataset_base', type=str, help='Path to root data folder.', default='/users/cost/jlovonme/data/hierarchy_props')
    p.add_argument('--property', type=int, help='Filter property or not.', default=-1)
    return p.parse_args()


if __name__ == "__main__":

    PARAMS = parse_args()
    print(PARAMS)

    train_model()


# Usage example

# python -u train_model.py --pool ${pool[$SLURM_ARRAY_TASK_ID]} --model_str
# /gpfsdswork/dataset/HuggingFace_Models/${model[$SLURM_ARRAY_TASK_ID]} --lr ${lrs[$SLURM_ARRAY_TASK_ID]} --bs 32
# --epochs 10 --output_dir /gpfsscratch/rech/evd/uwe77wt/hpatterns/${pool[$SLURM_ARRAY_TASK_ID]}-${lrs[$SLURM_ARRAY_TASK_ID]}-${names[$SLURM_ARRAY_TASK_ID]}-hp-v2
# --dataset_base /gpfswork/rech/evd/uwe77wt/data/hierarchy_props --property -1
