import argparse
import logging
from evaluation.extract_representations import get_dataset, get_embeddings_from_model

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')

def parse_args():
    p = argparse.ArgumentParser(description='Extract representation from models.', add_help=False)
    p.add_argument('--dataset_base', type=str, help='Path to root data folder.', default="data_generator/properties_v0")
    p.add_argument('--input', type=str, help='Path to the input data.', default="bansal_with_defs_test.json")
    p.add_argument('--out_dir', type=str, help='Output directory.', default='output/model_representations')
    
    p.add_argument('--model_name', type=str, help='Model name', default=None)
    p.add_argument('--model_source', type=str, help='Models path', default=None)
    p.add_argument('--input_type', type=str, help='Input types', default='concept') # concept or name
    
    p.add_argument('--only_last_layer', type=bool, help='Use only last layer?', default=False)

    return p.parse_args()


if __name__ == '__main__':
    
    params = parse_args()
    print(params)
    
    _dataset_base = params.dataset_base
    _ifile = params.input # only file needed for evaluation
    _output_name = params.out_dir
    model_source = params.model_source
    model_name = params.model_name

    if params.model_name is None: # If not names are given, use from the source
        model_name = params.model_source
        
    input_type = params.input_type
    only_last_layer = params.only_last_layer # True for only last layer, False to recover all layers

    logging.info("Reading dataset.")
    _dataset = get_dataset(_ifile, _dataset_base)

    # for model_str, model_src in zip(models_names, models_sources):
    #     for finput in input_list:
    logging.info(f"Processing model: {model_name} with file: {_ifile}.")
    get_embeddings_from_model(_dataset, model_str=model_name,model_src=model_source,
                              column=input_type,output_path=_output_name,only_last_layer=only_last_layer)


    # # For FastText
    # sources = ['embeddings/cc.en.300.bin','embeddings/wiki.en.bin']
    # names = ['cc','wiki']
    # for source,name in zip(sources,names):
    #     for finput in input_list:
    #         print(source, finput)
    #         get_embeddings_from_fasttext(_dataset, model_str='ft-'+name,source=os.path.join(_dataset_base,source), column=finput,output_path=_output_name)

    


#    #models_list = ['bert-base-cased','bert-large-cased','roberta-base','roberta-large','sentence-transformers/all-mpnet-base-v2']
    #models_list = ['sentence-transformers/all-mpnet-base-v2']
    #models_list = ['allenai/scibert_scivocab_cased']
    # models_list = ['ctp','ernie-base'] # 
    # models_list = ['sentence-transformers/bert-base-nli-mean-tokens','sentence-transformers/all-distilroberta-v1']

# Run example

# python run_extract_representations.py --model_name bert --model_source bert-base-cased \
# --only_last_layer True \
# --input_type concept \
# --dataset_base /home/jlovon/Datasets/hierarchy_props

# Or to save with default model_source name
# python run_extract_representations.py  --model_source bert-base-cased \
# --only_last_layer True \
# --input_type concept \
# --dataset_base /home/jlovon/Datasets/hierarchy_props