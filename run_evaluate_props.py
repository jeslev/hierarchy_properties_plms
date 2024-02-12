import argparse
from evaluation.evaluate_properties import get_results, get_table_results, get_random, examine_results
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description='Extract representation from models.', add_help=False)
    p.add_argument('--dataset_base', type=str, help='Path to root data folder.', default="data_generator/properties_v0")
    p.add_argument('--input', type=str, help='Path to the input data.', default="bansal_with_defs_test.json")
    p.add_argument('--vector_path', type=str, help='Folder where representations are stored .', default='output/model_representations')
    p.add_argument('--out_dir', type=str, help='Output directory.', default='output/props_scores')
    
    p.add_argument('--model_name', type=str, help='Model name', default=None)
    p.add_argument('--input_type', type=str, help='Input type to evaluate (name or concept)', default='concept')
    return p.parse_args()


def get_latex_results():

    models_name = ['BERT', 'BERT-L', 'RoBERTa', 'RoBERTa-L', 'S-BERT', 'DistilRoB', 'MPNet', 'ERNIE', 'CTP', 'SciBERT']
    tag_rep = 'cls' #avg_iso is without special tokens, avg_all is all tokens, cls is cls token
    sum_df = get_table_results(_models=models_name,basepath='output/props_scores', tag_rep=tag_rep, props=9)

    #sum_df['model'] = ['FT-cc','FT-wiki','BERT','BERT-L','RoBERTa','RoBERTa-L','MPNet','Scibert','CTP','ERNIE','S-BERT','DistilRoB']
    # sum_df['P'] = np.round( (ds_sizes[0]*sum_df['p1']+ds_sizes[1]*sum_df['p2']+ds_sizes[2]*sum_df['p3']+ds_sizes[3]*sum_df['p4'])/(ds_sizes[0]+ds_sizes[1]+ds_sizes[2]+ds_sizes[3]),1)
    # sum_df['A'] = np.round( (ds_sizes[4]*sum_df['p5']+ds_sizes[5]*sum_df['p6']+ds_sizes[6]*sum_df['p7'])/(ds_sizes[4]+ds_sizes[5]+ds_sizes[6]),1)
    # sum_df['S'] =np.round( (ds_sizes[7]*sum_df['p8']+ds_sizes[8]*sum_df['p9'])/(ds_sizes[7]+ds_sizes[8]),1)

    # Changed by 6 properties
    # sum_df['P'] = np.round( (sum_df['p1']+sum_df['p2']+sum_df['p3']+sum_df['p4'])/(4),1)
    # sum_df['A'] = np.round( (sum_df['p5']+sum_df['p6']+sum_df['p7'])/(3),1)
    # sum_df['S'] =np.round( (sum_df['p8']+sum_df['p9'])/(2),1)
    # sum_df['all'] =np.round( (sum_df['p1']+sum_df['p2']+sum_df['p3']+sum_df['p4']+
    #                         sum_df['p5']+sum_df['p6']+sum_df['p7']+
    #                           sum_df['p8']+sum_df['p9'])/(9),1)
    sum_df['P'] = np.round( (sum_df['p1']+sum_df['p2']+sum_df['p3'])/(3),1)
    sum_df['A'] = np.round( (sum_df['p5']+sum_df['p6'])/(2),1)
    sum_df['S'] =np.round( (sum_df['p8'])/(1),1)
    sum_df['all'] =np.round( (sum_df['p1']+sum_df['p2']+sum_df['p3']+
                            sum_df['p5']+sum_df['p6']+sum_df['p8'])/(6),1)
    print(sum_df.to_latex())


if __name__ == '__main__':
    # get_latex_results()
    # exit(0)

    params = parse_args()

    _dataset_base = params.dataset_base
    _itype = params.input_type
    _model = params.model_name
    vector_path = params.vector_path
    out_dir = params.out_dir

    
    _methods = ['cos','cos_l1','cosl2','cos_l3','euc','euc_l1','euc_l2','euc_l3']
    _models = ['ft-cc','ft-wiki','bert-base-cased','bert-large-cased','roberta-base','roberta-large','sentence-transformers/all-mpnet-base-v2',
          #'allenai/scibert_scivocab_cased','ctp',
           'ernie-base','sentence-transformers/bert-base-nli-mean-tokens','sentence-transformers/all-distilroberta-v1']

    
    #For qualitative analysis, commment to rerun distances
    
    # All setups    
#     _methods = ['cls','avg_a','cls_l1','avg_l1','cls_l2','avg_l2','cls_l3','avg_l3','ecls','eavg','ecls_l1','eavg_l1','ecls_l2','eavg_l2','ecls_l3','eavg_l3']
#     _methods_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#     _models = ['bert-base-cased', 'ctp','ernie-base','sentence-transformers/bert-base-nli-mean-tokens','sentence-transformers/all-distilroberta-v1']
#     _tag_reps = ['cls','avg_all','avg_iso']
    
    
    # Specific setup after analysis
    _inputtype = ['concept'] # word or concept
    _tag_reps = ['cls','avg_all']
    #_model = 'bert-base-cased'#, 'ctp','ernie-base','sentence-transformers/bert-base-nli-mean-tokens','sentence-transformers/all-distilroberta-v1']
    _models = ['bert_hierarchy','rob_hierarchy','drob_hierarchy', 'mpnet_hierarchy']
    _props = 9

    #for _method_id,_method in enumerate(_methods): # this is an ineficient approach, better try the one below
    for _model in _models:
        for _prop in range(_props):
            # Uncomment the following line to obtain random results
            for tag_rep in _tag_reps:
                get_results(_itype,_prop, _model, _dataset_base, tag_rep,
                                vector_path=vector_path)

                # uncomment to run qualitative anaylsis, if re-run expre, uncomment this line and comment the following one
                examine_results(_itype,_prop, _model, tag_rep,_dataset_base)

    # Random scores
#     for _prop in range(_props):
#         print(_prop)
#         get_random(_itype,_prop, _dataset_base, tries=10)
#         continue

#         for tag_rep in ['cls','avg_all','avg_iso']:

#             get_results(_itype,_prop, _methods,_model, _dataset_base, tag_rep,
#                         vector_path=vector_path, out_dir=out_dir)

            
# Scores per property after running 10 random runs
#  1  , 2,  3,  4, 5,  6,  7,  8,  9
# 50.54 49.8 50.03 50.11 50.12 50.21 50.52 50.38 49.84
