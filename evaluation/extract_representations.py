import os
import torch
import pandas as pd
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel, BertTokenizer, \
                         BertConfig, RobertaForSequenceClassification,RobertaTokenizer

import fasttext
from utils.utils import load_json, save_object, get_device, batch_to_device


def get_vocab_from_jsondf(inputfile, basepath):
    """
    Input:
        inputfile: Input file path
        Basepath: Data folder
    Returns:
        lvocab: List of vocabulary in the dataset
    The function reads the dataframe from the inputfile. Then, it recovers the unique triplets
    (tree_id, node name, node definition) as a list.
    """

    df = pd.read_json(os.path.join(basepath,inputfile), orient="index")
    # Iterate by subtrees, create directed graph + collect definitions
    all_vocab = {} # key:value -> (id_tree): {definitions:{}, graph:}
    for tid in range(min(df["treeid"]), max(df["treeid"])+1):
        # For each subtree
        filtered_df = df[df["treeid"]==tid]
        #print(display(filtered_df))
        # Retrieve definitions in this subtree
        for idx,row in filtered_df.iterrows():
            all_vocab[str(tid)+"-"+row['father']]=[tid,row['father'],row['father_definition']]
            all_vocab[str(tid)+"-"+row['child']]=[tid, row['child'],row['child_definition']]

        #subtrees[tid] = {'definitions':dictionary_defs, 'graph':G}
        #print("Root", [n for n,d in G.in_degree() if d==0] )
    lvocab =[]
    for k,v in all_vocab.items():
        lvocab.append(v)
    
    return lvocab


def get_dataset(inputfile, dataset_path):
    """
    Input:
        inputfile: json file
        dataset_path: path for the data folder

    Returns:
        Dataset (from HuggingFaces).
    It adds a new column with the constructed phrase with a word and its definition.
    """

    _all_vocab = get_vocab_from_jsondf(inputfile, dataset_path)
    df = pd.DataFrame(_all_vocab, columns=['tid','name','definition'])
    #df['name'] = df['name'].str.lower()
    #df['definition'] = df['definition'].str.lower()
    dataset = Dataset.from_pandas(df)
    dataset= dataset.add_column('concept',[n+" is defined as "+d for n,d in zip(dataset['name'],dataset['definition'])])
    #print("Loaded dataset")
    #print(dataset)
    return dataset


def get_model(model_str, model_src):

    if 'ERNIE' == model_str:
        config = BertConfig.from_pretrained('models/ernie_base/ernie_config.json')
        model = BertModel.from_pretrained('models/ernie_base',config=config) # For ERNIE
        tokenizer = BertTokenizer(vocab_file = 'models/ernie_base/vocab.txt', max_len=256,do_lower_case=False) # For ERNIE
    elif 'CTP' == model_str:
        model = RobertaForSequenceClassification.from_pretrained('roberta-large')
        model.load_state_dict(torch.load('/users/cost/jlovonme/ctp/outputs/ckpts/ckpt_par_wordnet_defs_roberta_large_1e6_seed2.json'))
        model = model.roberta
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large',max_len=256)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_src,do_lower_case=False)
        config = AutoConfig.from_pretrained(model_src, output_hidden_states=True)
        model = AutoModel.from_pretrained(model_src, config=config)

    return model, tokenizer

def get_embeddings_from_model(dataset, model_str='bert-base-uncased', model_src='bert-base-uncased',
                              column='name',output_path='.', only_last_layer=False):
    # type word: only name column, concept: name + definition
    # Load pretrained models
    
    model, tokenizer = get_model(model_str, model_src)
     
    # Preprocess data, tokenize dataset and load dataloader
    dataset = dataset.map(lambda e: tokenizer(e[column], truncation=True, padding='max_length'), batched=True)
    try: # bert based
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    except: # roberta
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1) # changed to 1 for scibert, default 32
    
    # identify if gpu
    device = get_device()
    # send model
    model.to(device)
    
    # Evaluate model
    model.eval() # set evaluation mode
    model_representations = {} # to store representations

    #return dataloader, model
    for batch_num, batch in enumerate(tqdm(dataloader)):
        #print(batch)
        batch = batch_to_device(batch,device)# send to device
        #print(batch.keys())
        b_input_mask = batch['attention_mask'] # to identify not null tokens 

        with torch.no_grad():
            out_features = model(**batch,
                    #input_ids=b_input_ids,
                    #token_type_ids=None,
                    #attention_mask=b_input_mask,
                    output_hidden_states=True)

        # This function modify model_representations
        compute_representations(out_features, b_input_mask, model_representations, only_last_layer=only_last_layer)
        #print("Layers",model_representations.keys(), "# of elements", len(model_representations[12][2][0]), "shape of mean fo 1 entry", model_representations[12][2][0][0].shape)
    #print(compute_representations())
    n_layers = list(model_representations.keys())
    print(n_layers)
    for layer in n_layers: # create dict per layer, each row [name,cls, avg_all,_]
        

        #all_representations = torch.cat(model_representations[layer],0) # put all batches in one dimension
        all_representations = model_representations[layer]  # cls, avg_all, avg
        print("Saving representations at layer", layer, " in total", len(all_representations), " .Shape 1 element", all_representations[0][0].shape)
        for vectors_rep, tag_rep in zip(all_representations,['cls','avg_all','avg_iso']):
            newtable = {}
            for row,name,tid in zip(vectors_rep,dataset['name'],dataset['tid']):
                #newline = [name,*row]
                newtable[(tid,name)] = row
            output_name = "global_test_{itype}_rep_{tag_rep}_{model}_layer_{layer}.pkl".format(itype=column,
                                                                                                model=model_str.replace('/','_'),
                                                                                                layer=str(layer),
                                                                                                tag_rep=tag_rep)
            save_object(newtable, os.path.join(output_path, output_name))
        


def compute_representations(out_features, b_input_mask, model_representations, only_last_layer=True):

    n_layer = 0
    if only_last_layer:
        total_layers = len(out_features['hidden_states'])
        n_layer = total_layers-1
        my_layers = [out_features['hidden_states'][-1]]
    else:
        my_layers = out_features['hidden_states']
        n_layer = 0

    for token_embeddings in my_layers: # iterate each layer with shape (batch_size, sequence_length, hidden_size)
        cls_rep = pool_cls_token(token_embeddings, b_input_mask).detach().cpu()
        avg_all_rep = pool_mean_token_special(token_embeddings, b_input_mask).detach().cpu()
        avg_rep = pool_mean_token(token_embeddings, b_input_mask).detach().cpu()

        representation_at_layer_x = model_representations.get(n_layer,[[],[],[]]) # get [cls, avg_all, avg]
        representation_at_layer_x[0].extend(cls_rep)
        representation_at_layer_x[1].extend(avg_all_rep)
        representation_at_layer_x[2].extend(avg_rep)

        model_representations[n_layer] = representation_at_layer_x
        n_layer += 1



    
def get_embeddings_from_fasttext(dataset, model_str='',source='', column='name',output_path='.'):
    model = fasttext.load_model(source)
    model_representations = {} # to store representations
    representation_at_layer_x = []
    for word in dataset[column]:
        sequence_representation = []
        
        for lword in word.split(" "): # Compute sentence embedding with average words
            sequence_representation.append(torch.Tensor(model.get_word_vector(lword)))
        
        sequence_representation = torch.mean(torch.stack(sequence_representation), axis=0)
        representation_at_layer_x.append([sequence_representation])
    model_representations[0] = representation_at_layer_x
    
    
    for layer in range(1): # create dict per layer, each row [name,cls, avg_all,_]
        print("Saving representations at layer", layer)
        newtable = {}
        for row,name,tid in zip(model_representations[layer],dataset['name'],dataset['tid']):
            newline = [name,*row]
            newtable[(tid,name)] = [*row]
        output_name = "global_test_{itype}_{model}_layer_{layer}.pkl".format(itype=column, model=model_str.replace('/','_'),layer=str(layer))
        save_object(newtable, os.path.join(output_path, output_name))

def pool_cls_token(token_embeddings, input_mask):
    # token_embeddings: [bs,seq,hid_dim], input_mask: [bs,seq]
    cls_tokens = token_embeddings[:,0,:] # Return first token
    assert len(token_embeddings) == len(input_mask)
    assert token_embeddings.shape[-1] == cls_tokens.shape[-1]

    return cls_tokens


def pool_mean_token_special(token_embeddings, input_mask):
    # token_embeddings: [bs,seq,hid_dim], input_mask: [bs,seq]

    # makes [bs,seq]->[bs,seq,hiddim]
    input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1) # total tokens
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return sum_embeddings / sum_mask

def pool_mean_token(token_embeddings, input_mask):

    # We need to get last mask = 1
     # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
    gather_indices = torch.argmin(input_mask, 1, keepdim=False) - 1  # Shape [bs]
    gather_indices = torch.clamp(gather_indices, min=0) # in case of empty sentences

    gather_indices = gather_indices.unsqueeze(1) # [bs] -> [bs,1]
    zero_mask = torch.zeros(token_embeddings.shape[0],1, dtype=int, device=get_device()) # [bs,1]

    input_mask = input_mask.scatter(1, gather_indices, zero_mask) # mask last token
    input_mask[:,0] = 0 # mask first token

    input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1) # total tokens
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return sum_embeddings / sum_mask



def get_embeddings_from_model_online(dataset, model_str='bert-base-uncased', model_src='bert-base-uncased',
                              only_last_layer=True):
    # dataset: dict[name:definition]
    # Load pretrained models
    model, tokenizer = get_model(model_str, model_src)

    # Preprocess data, tokenize dataset and load dataloader
    # prepare json
    new_json_dataset = {}
    list_keys = []
    list_defs = []
    list_concepts = []

    if type(dataset)==dict:

        for k,v in dataset.items():
            phrase = k +" is defined as "+v
            list_keys.append(k)
            list_defs.append(v)
            list_concepts.append(phrase)
            #list_keys.append(k)
            #new_json_dataset.append({"name":k, "definition":v, "concept": phrase})
    elif type(dataset) == list:
        for k in dataset:
            phrase = k
            list_keys.append(k)
            list_defs.append(k)
            list_concepts.append(k)

    new_json_dataset = {'name':list_keys, 'definition': list_defs, 'concept':list_concepts}
    dataset = Dataset.from_dict(new_json_dataset)
    dataset = dataset.map(lambda e: tokenizer(e['concept'], truncation=True, padding='max_length'), batched=True)
    try: # bert based
        dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    except: # roberta
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32) # changed to 1 for scibert, default 32

    # identify if gpu
    device = get_device()
    # send model
    model.to(device)

    # Evaluate model
    model.eval() # set evaluation mode
    model_representations = {} # to store representations

    #return dataloader, model
    for batch_num, batch in enumerate(tqdm(dataloader)):
        #print(batch)
        batch = batch_to_device(batch,device)# send to device
        #print(batch.keys())
        b_input_mask = batch['attention_mask'] # to identify not null tokens

        with torch.no_grad():
            out_features = model(**batch,
                    #input_ids=b_input_ids,
                    #token_type_ids=None,
                    #attention_mask=b_input_mask,
                    output_hidden_states=True)

        # This function modify model_representations, get [cls, avg_all,avg] in model_representations array
        compute_representations(out_features, b_input_mask, model_representations, only_last_layer=only_last_layer)
        #print("Layers",model_representations.keys(), "# of elements", len(model_representations[12][2][0]), "shape of mean fo 1 entry", model_representations[12][2][0][0].shape)
    #print(compute_representations())
    n_layers = list(model_representations.keys())
    #print("# layers",n_layers)
    overall_representations = {} # dictionary for cls,avg_all,avg_iso for each element in list_keys
    for layer in n_layers: # create dict per layer, each row [name,cls, avg_all,_]
        all_representations = model_representations[layer]  # cls, avg_all, avg
        #for vectors_rep, tag_rep in zip(all_representations,['cls','avg_all','avg_iso']):
        overall_representations['cls'] = all_representations[0]
        overall_representations['avg_all'] = all_representations[1]
        overall_representations['avg_iso'] = all_representations[2]
    return overall_representations, list_keys