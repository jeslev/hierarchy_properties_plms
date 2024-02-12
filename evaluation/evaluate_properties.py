from utils import load_object, load_json, save_json, euclidean_distance, cosinus_distance

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_distances(v1,v2,v3):
    """ method_id table:
    Methods = cosinus, +lear 1, + lear 2, + lear 3
    Method            | CLS |
    cosinus           |  0  |
    cosinues + lear_1 |  1  |
    cosinues + lear_2 |  2  |
    cosinues + lear_3 |  3  |
    euclid            |  4  |
    euclid + lear_1   |  5 |
    euclid + lear_2   |  6 |
    euclid + lear_3   |  7 |
    
    """
    cos_d1 = cosinus_distance(v1,v2)
    cos_d2 = cosinus_distance(v1,v3)
    euc_d1 = euclidean_distance(v1,v2)
    euc_d2 = euclidean_distance(v1,v3)
    
    n_v1 = torch.norm(v1)
    n_v2 = torch.norm(v2)
    n_v3 = torch.norm(v3)
    
    # Absolute valor LEAR distance
#     n_diff_norm = [ [abs(n_v1[0]-n_v2[0]),abs(n_v1[0]-n_v3[0])], [abs(n_v1[1]-n_v2[1]),abs(n_v1[1]-n_v3[1])] ] 
#     n_sum_norm = [ [n_v1[0]+n_v2[0],n_v1[0]+n_v3[0]], [n_v1[1]+n_v2[1],n_v1[1]+n_v3[1]] ] 
#     n_max_norm = [ [max(n_v1[0],n_v2[0]),max(n_v1[0],n_v3[0])], [max(n_v1[1],n_v2[1]),max(n_v1[1],n_v3[1])] ] 
    
    # Hypo X-Y (X is a child of Y)
#     n_diff_norm = [ [n_v1[0]-n_v2[0],n_v1[0]-n_v3[0]], [n_v1[1]-n_v2[1],n_v1[1]-n_v3[1]] ] 
#     n_sum_norm = [ [n_v1[0]+n_v2[0],n_v1[0]+n_v3[0]], [n_v1[1]+n_v2[1],n_v1[1]+n_v3[1]] ] 
#     n_max_norm = [ [max(n_v1[0],n_v2[0]),max(n_v1[0],n_v3[0])], [max(n_v1[1],n_v2[1]),max(n_v1[1],n_v3[1])] ] 
    
    # Hyper X-Y (X is a parent of Y)
    n_diff_norm = [ -(n_v1-n_v2),-(n_v1-n_v3)]
    n_sum_norm = [ n_v1+n_v2,n_v1+n_v3]
    n_max_norm = [ max(n_v1,n_v2),max(n_v1,n_v3)]

    distances_list_1, distances_list_2 = [], []
    distances_list_1.append(cos_d1)
    distances_list_1.append(cos_d1+n_diff_norm[0])
    distances_list_1.append(cos_d1+n_diff_norm[0]/n_sum_norm[0])
    distances_list_1.append(cos_d1+n_diff_norm[0]/n_max_norm[0])
    distances_list_1.append(euc_d1)
    distances_list_1.append(euc_d1+n_diff_norm[0])
    distances_list_1.append(euc_d1+n_diff_norm[0]/n_sum_norm[0])
    distances_list_1.append(euc_d1+n_diff_norm[0]/n_max_norm[0])

    distances_list_2.append(cos_d2)
    distances_list_2.append(cos_d2+n_diff_norm[1])
    distances_list_2.append(cos_d2+n_diff_norm[1]/n_sum_norm[1])
    distances_list_2.append(cos_d2+n_diff_norm[1]/n_max_norm[1])
    distances_list_2.append(euc_d2)
    distances_list_2.append(euc_d2+n_diff_norm[1])
    distances_list_2.append(euc_d2+n_diff_norm[1]/n_sum_norm[1])
    distances_list_2.append(euc_d2+n_diff_norm[1]/n_max_norm[1])

    return distances_list_1,distances_list_2

def get_results(input_type,prop,model, basedata, tag_rep, vector_path='model_vectors', out_dir='output/computed_results'):
    """
    Evaluate properties only for the last layer of a given model.
    """
    _propfile = "test_hard_neg.json"

    # Load and filter dataset
    df = pd.read_json(os.path.join(basedata,_propfile), orient="index")
    ndf = df[df['prop']==(prop+1)].drop_duplicates()

    bigtable = []
    layer_sizes = 32
    
    accepted_layers = set()
    for layer in range(0,layer_sizes+1): # Only want last layer, search for it
        model_rep = 'global_test_{inputtype}_rep_{tag_rep}_{model}_layer_{layer}.pkl'.format(inputtype=input_type,
                                                                               model=model.replace('/','_'),
                                                                               layer=str(layer),
                                                                               tag_rep=tag_rep)
        
        try:
            vectors = load_object(os.path.join(vector_path,model_rep))
            accepted_layers.add(layer)
            
        except:
            continue
    

    last_layer = max(list(accepted_layers))
    print("Initiate getting results with layer", last_layer, "representation method:", tag_rep, " for property ", prop)

    for layer in range(last_layer,last_layer+1):

        model_rep = 'global_test_{inputtype}_rep_{tag_rep}_{model}_layer_{layer}.pkl'.format(inputtype=input_type,
                                                                                      model=model.replace('/','_'),
                                                                                      layer=str(layer),
                                                                                      tag_rep=tag_rep)
        try:
            vectors = load_object(os.path.join(vector_path,model_rep))
        except:
            break # if not more layers go to next model
        result_row = [input_type,model,prop,layer]

        pbar = tqdm(ndf.iterrows())
        total = len(ndf)
        lcorrect = [0]*16#len(methods) # keep counter in each method
        
        for idx,row in pbar:
            pbar.set_description(f"Processing  {model} {layer} {prop}") 
            
            k1 = (row['tree_id'],row['ent_1'])
            k2 = (row['tree_id'],row['ent_2'])
            k3 = (row['tree_id'],row['ent_3'])

            v1,v2,v3 = vectors.get(k1,None),vectors.get(k2,None),vectors.get(k3,None)

            if None in [v1,v2,v3]:
                print("why?", k1,k2,k3)

            # compute result
            ld1,ld2 = compute_distances(v1,v2,v3) # list of both distances for all methods list
            
            method_id = 0
            for d1,d2 in zip(ld1,ld2):
                if d1<d2:
                    lcorrect[method_id] += 1
                method_id += 1
            
        result = [ round(correct/total*100,2) for correct in lcorrect ]
        result_row.extend(result)
        bigtable.append(result_row)
    
    # Write bigtable
    methods = ['cls','avg_a','cls_l1','avg_l1','cls_l2','avg_l2','cls_l3','avg_l3','ecls','eavg','ecls_l1','eavg_l1','ecls_l2','eavg_l2','ecls_l3','eavg_l3']
    df = pd.DataFrame(bigtable, columns=['input','model','prop','layer',*methods])
    data_json = df.to_json(orient="index")
    output_file = 'results_{input_type}_p{prop}_{model}_rep_{tag_rep}.json'.format(input_type=input_type,
                                                                            prop=prop,
                                                                            model=model.replace('/','_'),
                                                                            tag_rep=tag_rep)
    save_json(data_json, os.path.join(out_dir,output_file))


    
def examine_results(input_type,prop,model, tag_rep, basedata):

    _propfile = "data_generator/properties_v0/bansal_with_defs_testprop_{prop}.json".format(prop=str(prop))

    # Load and filter dataset
    df = pd.read_json(_propfile, orient="index")
    ndf = df[df['valid']==1]


    bigtable = []
    layer_sizes = 32
    lcorrect = []
    
    last_layer = 0
    for layer in range(0,layer_sizes+1): # Only want last layer, search for it
        model_rep = 'output/model_representations/global_test_{inputtype}_rep_{tag_rep}_{model}_layer_{layer}.pkl'.format(inputtype=input_type,
                                                                                      model=model.replace('/','_'),
                                                                                      layer=str(layer),
                                                                                      tag_rep=tag_rep)
        try:
            vectors = load_object(model_rep)
            last_layer = layer
        except:
            continue # if not more layers go to next model
    
    
    for layer in range(last_layer,last_layer+1): # Only process last layer, kept for because before we searched every layer

        model_rep = 'output/model_representations/global_test_{inputtype}_rep_{tag_rep}_{model}_layer_{layer}.pkl'.format(inputtype=input_type,
                                                                                      model=model.replace('/','_'),
                                                                                      layer=str(layer),
                                                                                      tag_rep=tag_rep)
        #print(model_rep)
        try:
            vectors = load_object(model_rep)
        except:
            break # if not more layers go to next model
        lcorrect = []

        pbar = tqdm(ndf.iterrows())
        total = len(ndf)
        
        
        for idx,row in pbar:
            pbar.set_description(f"Processing  {model} {layer} {prop}") 
            
            k1 = (row['tree_id'],row['ent_1'])
            k2 = (row['tree_id'],row['ent_2'])
            k3 = (row['tree_id'],row['ent_3'])

            v1,v2,v3 = vectors.get(k1,None),vectors.get(k2,None),vectors.get(k3,None)
            if model.startswith('ft-'):
                #print(len(v1))
                v1,v2,v3 = [v1[0],v1[0]],[v2[0],v2[0]],[v3[0],v3[0]]
            if None in [v1,v2,v3]:
                print("why?", k1,k2,k3)
            
            # compute result
            
            ld1,ld2 = compute_distances(v1,v2,v3) # list of both distances for all methods list
            # Get only avg_a metric
            d1 = ld1[1]
            d2 = ld2[1]
            
            if d1<d2:
                lcorrect.append(True)
            else:
                lcorrect.append(False)
    
    # Write bigtable
    #print(len(ndf), len(lcorrect))
    ndf['correct'] = lcorrect
    data_json = ndf.to_json(orient="index")
    save_json(data_json, 'output/computed_results/single_results_{input_type}_p{prop}_{model}.json'.format(input_type=input_type,prop=prop,model=model.replace('/','_')))



def create_table_results(itype, models,props=9, tag_rep='cls',basepath='.'):
    bigtable = []
    _props = 9
    for _prop in range(props):
        for _model in models:
            _ifile = os.path.join(basepath,'results_{input_type}_p{prop}_{model}_rep_{tag_rep}.json'.format(input_type=itype,
                                                                                                   prop=_prop,
                                                                                                   model=_model.replace('/','_'),
                                                                                                    tag_rep=tag_rep))
            df = pd.read_json(load_json(_ifile), orient="index")
            bigtable.append(df)
    bigdf = pd.concat(bigtable)
    return bigdf


def get_last_layers(bigdf):
    # Filter only last layer for each model
    maxlayer_dict = {}
    for (idx,row) in bigdf.groupby(['model']).max()['layer'].iteritems():
        maxlayer_dict[idx] = row


    last_layers = []
    for mod,lay in maxlayer_dict.items():
        last_layers.append(bigdf[(bigdf['model']==mod) & (bigdf['layer']==lay)])

    lastlayer_df = pd.concat(last_layers)
    return lastlayer_df

def get_table_results(_models=None, basepath=None, tag_rep='cls',props=9):
    # Table in Latex to summarize the score on each property of the models
    # We chose concept input, avg method, cosinus distances without LEAR, last layer/
    # Table in paper

    if _models is None:
        _models = ['ft-cc','ft-wiki','bert-base-cased','bert-large-cased','roberta-base','roberta-large','sentence-transformers/all-mpnet-base-v2',
          'allenai/scibert_scivocab_cased','ctp','ernie-base','sentence-transformers/bert-base-nli-mean-tokens','sentence-transformers/all-distilroberta-v1']

    bigdf = create_table_results('concept', _models, props=props, tag_rep=tag_rep,basepath=basepath)
    _lastlayers = get_last_layers(bigdf)

    summary_table = []
    for model in _models:
        row = [model]
        minidf = _lastlayers[_lastlayers['model']==model]
        # Change property computation
        # for prop in range(props):
        #     score = minidf[minidf['prop']==prop]['cos']
        #     row.append(round(float(score),1))
        score = minidf[minidf['prop']==0]['cos'] # prop 1
        row.append(round(float(score),1))
        score = minidf[minidf['prop']==1]['cos'] # prop 2
        row.append(round(float(score),1))
        score = minidf[minidf['prop']==2]['cos'] # prop 3/4 -> prop 3
        score2 = minidf[minidf['prop']==3]['cos']
        row.append(round(float(score+score2)/2.0,1))
        score = minidf[minidf['prop']==4]['cos'] # prop 5
        row.append(round(float(score),1))
        score = minidf[minidf['prop']==5]['cos'] # prop 6/7 -> prop 6
        score2 = minidf[minidf['prop']==6]['cos']
        row.append(round(float(score+score2)/2.0,1))
        score = minidf[minidf['prop']==7]['cos'] # prop 8/9 -> prop 8
        score2 = minidf[minidf['prop']==8]['cos']
        row.append(round(float(score+score2)/2.0,1))

        summary_table.append(row)
    #_lastlayers[['input','model','prop','layer','cls']]

    #sum_df = pd.DataFrame(summary_table, columns=['model','p1','p2','p3','p4','p5','p6','p7','p8','p9'])
    sum_df = pd.DataFrame(summary_table, columns=['model','p1','p2','p3','p5','p6','p8'])
    return sum_df



def get_random(input_type,prop, basedata, tries=5):
    """
    Evaluate properties only for the last layer of a given model.
    """
    _propfile = "test_hard_neg.json"

    # Load and filter dataset
    df = pd.read_json(os.path.join(basedata,_propfile), orient="index")
    ndf = df[df['prop']==(prop+1)].drop_duplicates()

    total_acc = 0.0
    dict_entities, rev_dict = {}, {}

    for tree_id in set(ndf['tree_id']):
        subdf = ndf[ndf['tree_id']==tree_id]
        all_entities = set(subdf['ent_1']).union(set(subdf['ent_2'])).union(set(subdf['ent_3']))
        #print(tree_id, len(all_entities))
        for idx,ent in enumerate(all_entities):
            dict_entities[(tree_id,ent)] = idx
            rev_dict[str(tree_id)+"-"+str(idx)] = (tree_id,ent)

    avg_acc = 0.0
    for run in range(tries):
        distances = {}
        tot_correct = 0 # nb of correct prediction on this dataframe
        total = 0

        # Create symmetric distance matrix per tree_id
        for tree_id in set(ndf['tree_id']):
            subdf = ndf[ndf['tree_id']==tree_id]
            all_entities = set(subdf['ent_1']).union(set(subdf['ent_2'])).union(set(subdf['ent_3']))
            N = len(all_entities) # matrix size
            b = np.random.random(size=(N,N))
            distances[tree_id] = (b + b.T)/2

        # Evaluates dataframe
        for idx,row in ndf.iterrows():
            idx1 = dict_entities[(row['tree_id'],row['ent_1'])] #(row['tree_id'],row['ent_1'])
            idx2 = dict_entities[(row['tree_id'],row['ent_2'])]
            idx3 = dict_entities[(row['tree_id'],row['ent_3'])]
            #print(distances[row['tree_id']])
            #print(row['tree_id'],idx1, idx2, idx3)
            d1 = distances[row['tree_id']][idx1,idx2]
            d2 = distances[row['tree_id']][idx1,idx3]
            correct = 1 if d1<d2 else 0
            tot_correct += correct
            total += 1

        total_acc = (tot_correct / total) # accuracy dataframe
        #print(tot_correct, total,(tot_correct / float(total)))
        avg_acc += total_acc
    #print(avg_acc, tries, round(avg_acc/float(tries)*100,2))
    print( round(avg_acc/float(tries)*100,2))
