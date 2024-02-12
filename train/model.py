from sentence_transformers import models, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers import SentenceTransformer, InputExample

import torch
from torch.utils.data import DataLoader

from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from typing import List

import math
import networkx as nx
from tqdm import tqdm
from tqdm.autonotebook import trange
import numpy as np

from utils.utils import batch_to_device, get_device, cosinus_distance
from evaluation.extract_representations import pool_cls_token, pool_mean_token, pool_mean_token_special

logger = logging.getLogger(__name__)


class Model:

    def __init__(self, max_len=256, lower_case=False, batch_size=8, epochs=10,
                evaluation_steps=3000, learning_rate=1e-5, output_dir = '.'):
        super()
        self.max_len = max_len
        self.lower_case = lower_case
        self.batch_size = batch_size
        self.evaluation_steps=evaluation_steps
        self.epochs = epochs
        self.lr = learning_rate
        self.model = None
        self.train_loss = None
        self.optimizer = None
        self.mode = None
        self.output_dir = output_dir
        self.cur_epoch = 0

    def create_model_sentencetransformer(self, model_str='bert-base-cased',pooling_mode='mean', optimizer='AdamW'):
        """
        Simple transformer model with Pooling Layer (mean pooling or cls) at the end.
        """

        tokenizer_max_len=self.max_len
        word_embedding_model = models.Transformer(model_str, do_lower_case=self.lower_case, max_seq_length=tokenizer_max_len)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)

        self.mode = pooling_mode
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        self.train_loss = losses.TripletLoss(model=self.model,distance_metric=losses.TripletDistanceMetric.COSINE)#loss(model=model)
        self.optimizer = eval('torch.optim.'+optimizer)

    def get_dataloader(self, train_samples, shuffle=True):
        return DataLoader(train_samples, shuffle=shuffle, batch_size=self.batch_size)

    def set_triplet_evaluator(self, dev_samples, name='dev'):
        self.evaluator = TripletEvaluator.from_input_examples(dev_samples, name=name,
                                                              main_distance_function=SimilarityFunction.COSINE,
                                                             show_progress_bar=False)

    def set_taxonomy_evaluator(self, dev_samples, name='dev'):
        self.evaluator = TaxonomyEvaluator.from_input_examples(dev_samples, name=name)

    def set_triplet_evaluator_hf(self, dev_samples, name='dev'):
        self.evaluator = TripletEvaluator_HF.from_input_examples(dev_samples, name=name)

    def set_triplet_filtered_evaluator(self, dev_samples, test_samples, name='dev'):
        self.evaluator = TripletEvaluator_filtered.from_input_examples(dev_samples, name=name, mode=self.mode)
        self.evaluator_test_mean = TripletEvaluator_filtered.from_input_examples(test_samples, name='test_mean', mode='mean')
        self.evaluator_test_cls = TripletEvaluator_filtered.from_input_examples(test_samples, name='test_cls', mode='cls')

    def train(self, train_loader, get_scores, warmup_ratio):
        warmup_steps = math.ceil(len(train_loader) * self.epochs * warmup_ratio) #10% of train data for warm-up
        # Train the model
        self.model.fit(train_objectives=[(train_loader, self.train_loss)],
                  evaluator=self.evaluator,
                  epochs=self.epochs,
                  evaluation_steps=self.evaluation_steps,#int(len(loader)*0.1),
                  warmup_steps=warmup_steps,
                  #output_path=output_dir, # not saving for now
                  #save_best_model=True,
                  optimizer_class=self.optimizer,
                  optimizer_params={'lr':self.lr},
                  callback=get_scores,
                  #use_amp=False          #Set to True, if your GPU supports FP16 operations
                  )


    def train_and_test(self, train_loader, warmup_ratio):

        def run_test(accuracy, epoch, steps):
            if steps==-1: # after epoch
                return_sth_mean = self.model.evaluate(self.evaluator_test_mean, output_path=self.output_dir)
                return_sth_cls = self.model.evaluate(self.evaluator_test_cls, output_path=self.output_dir)
                self.cur_epoch += 1
                self.model.save(os.path.join(self.output_dir,"epoch"+str(self.cur_epoch)))
                print("Return from test mean", return_sth_mean)
                print("Return from test cls", return_sth_cls)

        warmup_steps = math.ceil(len(train_loader) * self.epochs * warmup_ratio) #10% of train data for warm-up

        # zero-shot evaluation of the model
        sth_mean = self.model.evaluate(self.evaluator_test_mean, output_path=self.output_dir)
        sth_cls = self.model.evaluate(self.evaluator_test_cls, output_path=self.output_dir)
        print("Return from test mean", sth_mean)
        print("Return from test cls", sth_cls)

        # Train the model
        self.model.fit(train_objectives=[(train_loader, self.train_loss)],
                  evaluator=self.evaluator,
                  epochs=self.epochs,
                  evaluation_steps=self.evaluation_steps,#int(len(loader)*0.1),
                  warmup_steps=warmup_steps,
                  output_path=self.output_dir,
                  save_best_model=True,
                  optimizer_class=self.optimizer,
                  optimizer_params={'lr':self.lr},
                  callback=run_test,
                  #use_amp=False          #Set to True, if your GPU supports FP16 operations
                  )

        #Save model
        model_path = os.path.join(self.output_dir,"final")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.model.save(model_path)



class TaxonomyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a taxonomy prediction: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
    """

    def __init__(
        self,
        examples,
        main_distance_function: SimilarityFunction = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param negatives: List of negative sentences
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.examples = examples
        self.name = name

        #assert len(self.anchors) == len(self.positives)
        #assert len(self.anchors) == len(self.negatives)

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhattan", "accuracy_euclidean"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples, **kwargs):

        return cls(examples, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TaxonomyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        tokenizer = model.tokenizer
        transformer = model[0].auto_model # without auto_model we use the Transformer class from Sentence Transformer that it's different from HFs
        total_subtrees = len(self.examples)
        # identify if gpu and send model
        device = get_device()
        transformer.to(device)
        transformer.eval() # set evaluation mode

        total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0

        for tree_id,[golden_edges, definitions] in tqdm(self.examples.items()):

            # Create golden graph
            gold_tree = nx.Graph()
            for golden_edge in golden_edges:
                gold_tree.add_edge(*golden_edge)
            #root_node = list(nx.topological_sort(gold_tree))[0]
            #print("root", root_node, gold_tree.edges())
            # Prepare definitions
            inputs = []
            keys = []
            for k,definition in definitions.items():
                keys.append(k)
                inputs.append(definition)

            # Obtain representations
            # Preprocess data, tokenize dataset and load dataloader
            model_representations = [] # to store representations
            #print(transformer)
            for start_index in trange(0, len(inputs), self.batch_size, desc="Batches", disable=True):
                batch = inputs[start_index:start_index+self.batch_size]
                batch = tokenizer(batch, truncation=True, padding='max_length',return_tensors='pt')
                batch = batch_to_device(batch,device)# send to device
                b_input_mask = batch['attention_mask'] # to identify not null tokens

                with torch.no_grad():
                    out_features = transformer(**batch,
                            output_hidden_states=True)

                my_layers = [out_features['hidden_states'][-1]]

                for token_embeddings in my_layers: # iterate each layer with shape (batch_size, sequence_length, hidden_size)
                    # These pooling gives back [bs,hid_dim]
                    cls_rep = pool_cls_token(token_embeddings, b_input_mask).detach().cpu()
                    #avg_all_rep = pool_mean_token_special(token_embeddings, b_input_mask).detach().cpu()
                    #avg_rep = pool_mean_token(token_embeddings, b_input_mask).detach().cpu()
                    model_representations.extend(cls_rep)

            # Create full graph
            count = 0
            G = nx.Graph()
            for k,avg_rep in zip(keys,model_representations):
                for k2,avg_rep2 in zip(keys[count+1:],model_representations[count+1:]):
                    dist = cosinus_distance(avg_rep, avg_rep2)
                    G.add_edge(k,k2,weight=dist) # to full graph
                count+=1

            # Do MST
            alg = nx.algorithms.tree.Edmonds(G)
            predicted_tree = alg.find_optimum(style="arborescence")

            #print(predicted_tree.edges())
            #print(gold_tree.edges())
            # Compare tree
            precision, recall, f1 = compute_edge_metrics(predicted_tree=predicted_tree,
            gold_tree=gold_tree)

            # Compute metrics
            #print(precision, recall, f1)
            total_precision += precision
            total_recall += recall
            total_f1 += f1

        total_precision = total_precision /(total_subtrees*1.0)
        total_recall = total_recall /(total_subtrees*1.0)
        total_f1 = total_f1 /(total_subtrees*1.0)

        logger.info("Precision:   \t{:.2f}".format(total_precision * 100))
        logger.info("Recall:\t{:.2f}".format(total_recall * 100))
        logger.info("F1:\t{:.2f}\n".format(total_f1 * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, total_precision, total_recall, total_f1])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, total_precision, total_recall, total_f1])
        return total_f1


class TripletEvaluator_HF(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        main_distance_function: SimilarityFunction = None,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param negatives: List of negative sentences
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhattan", "accuracy_euclidean"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        
        logger.info("TripletEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        # Get transformer model
        tokenizer = model.tokenizer
        transformer = model[0].auto_model

        # identify if gpu
        device = get_device()
        transformer.to(device)
        transformer.eval() # set evaluation mode

        num_triplets = 0
        num_correct_cos_triplets, num_correct_manhattan_triplets, num_correct_euclidean_triplets = 0, 0, 0

        total_samples = len(self.anchors)
        embeddings_anchors = self.get_representations(total_samples, self.anchors, tokenizer, transformer, device)
        embeddings_positives = self.get_representations(total_samples, self.positives, tokenizer, transformer, device)
        embeddings_negatives = self.get_representations(total_samples, self.negatives, tokenizer, transformer, device)

        # Cosine distance
        pos_cos_distance = paired_cosine_distances(embeddings_anchors, embeddings_positives)
        neg_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_negatives)

        # Manhattan
        pos_manhattan_distance = paired_manhattan_distances(embeddings_anchors, embeddings_positives)
        neg_manhattan_distances = paired_manhattan_distances(embeddings_anchors, embeddings_negatives)

        # Euclidean
        pos_euclidean_distance = paired_euclidean_distances(embeddings_anchors, embeddings_positives)
        neg_euclidean_distances = paired_euclidean_distances(embeddings_anchors, embeddings_negatives)

        for idx in range(len(pos_cos_distance)):
            num_triplets += 1

            if pos_cos_distance[idx] < neg_cos_distances[idx]:
                num_correct_cos_triplets += 1

            if pos_manhattan_distance[idx] < neg_manhattan_distances[idx]:
                num_correct_manhattan_triplets += 1

            if pos_euclidean_distance[idx] < neg_euclidean_distances[idx]:
                num_correct_euclidean_triplets += 1

        accuracy_cos = num_correct_cos_triplets / num_triplets
        accuracy_manhattan = num_correct_manhattan_triplets / num_triplets
        accuracy_euclidean = num_correct_euclidean_triplets / num_triplets

        logger.info("Accuracy Cosine Distance:   \t{:.2f}".format(accuracy_cos * 100))
        logger.info("Accuracy Manhattan Distance:\t{:.2f}".format(accuracy_manhattan * 100))
        logger.info("Accuracy Euclidean Distance:\t{:.2f}\n".format(accuracy_euclidean * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

        return accuracy_cos
        if self.main_distance_function == SimilarityFunction.COSINE:
            return accuracy_cos
        if self.main_distance_function == SimilarityFunction.MANHATTAN:
            return accuracy_manhattan
        if self.main_distance_function == SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        return max(accuracy_cos, accuracy_manhattan, accuracy_euclidean)


    def get_representations(self, total_samples, sentences, tokenizer, transformer, device):
        model_representations = []
        
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        
        for start_index in trange(0, total_samples, self.batch_size, desc="Batches", disable=True):
            batch = sentences_sorted[start_index:start_index+self.batch_size]
            batch = tokenizer(batch, padding=True,return_tensors='pt')
            batch = batch_to_device(batch,device)# send to device
            b_input_mask = batch['attention_mask'] # to identify not null tokens

            with torch.no_grad():
                    out_features = transformer(**batch,
                            output_hidden_states=True)

            token_embeddings = out_features['hidden_states'][-1] # last_layer
            avg_rep = pool_mean_token(token_embeddings, b_input_mask).detach().cpu()            
            model_representations.extend(avg_rep)
            #for token_embeddings in my_layers: # iterate each layer with shape (batch_size, sequence_length, hidden_size)
                    # These pooling gives back [bs,hid_dim]
                    #cls_rep = pool_cls_token(token_embeddings, b_input_mask).detach().cpu()
                    #avg_all_rep = pool_mean_token_special(token_embeddings, b_input_mask).detach().cpu()
             #       avg_rep = pool_mean_token(token_embeddings, b_input_mask).detach().cpu()
        
        model_representations = [model_representations[idx] for idx in np.argsort(length_sorted_idx)]
        model_representations = np.asarray([emb.numpy() for emb in model_representations])
        return model_representations
    
    
    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings
    

class TripletEvaluator_filtered(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
    """

    def __init__(
        self,
        anchors,
        positives,
        negatives,
        main_distance_function: SimilarityFunction = None,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        mode: str = 'cls',
    ):
        """
        :param anchors: Dictionary of sentences to check similarity to. (e.g. a query)
        :param positives: Dictionary of sentences with List of positive sentences
        :param negatives: Dictionary of sentences with List of negative sentences
        :param main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        :param mode: which mode to pool fro model cls or mean
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name
        self.mode = mode # cls or mean

        assert len(self.anchors.keys()) == len(self.positives.keys())
        assert len(self.anchors.keys()) == len(self.negatives.keys())

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_" + mode + "_results.csv"
        self.csv_headers = ["epoch", "steps", "p1", "p2","p3","p4","p5","p6","p7","p8","p9","P-1","P-2","P-3","A-1",
                            "A-2","S-1","P","A","S","all"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[List], **kwargs):
        anchors = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[]}
        positives = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[]}
        negatives = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[]}

        for prop_name in ['p1','p2','p3','p4','p5','p6','p7','p8','p9']:
            for example in examples[prop_name]:
                for el_id, dict_prop in zip([0,1,2],[anchors, positives, negatives]):
                    old_list = dict_prop.get(prop_name)
                    old_list.append(example.texts[el_id])
                    dict_prop[prop_name] = old_list
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TripletEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        # Get transformer model
        tokenizer = model.tokenizer
        transformer = model[0].auto_model

        # identify if gpu
        device = get_device()
        transformer.to(device)
        transformer.eval() # set evaluation mode

        all_scores = []
        for single_prop in ['p1','p2','p3','p4','p5','p6','p7','p8','p9']:
            num_triplets = 0
            num_correct_cos_triplets = 0

            total_samples = len(self.anchors[single_prop])
            embeddings_anchors = self.get_representations(total_samples, self.anchors[single_prop], tokenizer, transformer, device)
            embeddings_positives = self.get_representations(total_samples, self.positives[single_prop], tokenizer, transformer, device)
            embeddings_negatives = self.get_representations(total_samples, self.negatives[single_prop], tokenizer, transformer, device)

            # Cosine distance
            pos_cos_distance = paired_cosine_distances(embeddings_anchors, embeddings_positives)
            neg_cos_distances = paired_cosine_distances(embeddings_anchors, embeddings_negatives)

            for idx in range(len(pos_cos_distance)):
                num_triplets += 1
                if pos_cos_distance[idx] < neg_cos_distances[idx]:
                    num_correct_cos_triplets += 1


            accuracy_cos = num_correct_cos_triplets / num_triplets
            all_scores.append(accuracy_cos)

        logger.info("Accuracy Cosine Distance for Properties:",[round(score*100,2) for score in all_scores])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            to_writerow = [epoch, steps]
            for score in all_scores:
                to_writerow.append(round(score*100,2))
            # Add aggregated scores
            p1=all_scores[0]
            p2=all_scores[1]
            p3=(all_scores[2]+all_scores[3])/2.0
            a1=all_scores[4]
            a2=(all_scores[5]+all_scores[6])/2.0
            s1=(all_scores[7]+all_scores[8])/2.0
            p=(p1+p2+p3)/3.0
            a=(a1+a2)/2.0
            s=s1
            t=(p+a+s)/3.0
            for x in [p1,p2,p3,a1,a2,s1,p,a,s,t]:
                to_writerow.append(round(x*100,2)) 

            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(to_writerow)
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(to_writerow)

        return t

    def get_representations(self, total_samples, sentences, tokenizer, transformer, device):
        model_representations = []

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, total_samples, self.batch_size, desc="Batches", disable=True):
            batch = sentences_sorted[start_index:start_index+self.batch_size]
            batch = tokenizer(batch, padding=True,return_tensors='pt')
            batch = batch_to_device(batch,device)# send to device
            b_input_mask = batch['attention_mask'] # to identify not null tokens

            with torch.no_grad():
                    out_features = transformer(**batch,
                            output_hidden_states=True)

            token_embeddings = out_features['hidden_states'][-1] # last_layer
            if self.mode =='cls':
                _rep = pool_cls_token(token_embeddings, b_input_mask).detach().cpu()
            elif self.mode == 'mean':
                _rep = pool_mean_token_special(token_embeddings, b_input_mask).detach().cpu()

            model_representations.extend(_rep)

        model_representations = [model_representations[idx] for idx in np.argsort(length_sorted_idx)]
        model_representations = np.asarray([emb.numpy() for emb in model_representations])
        return model_representations


    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings


def compute_edge_metrics(predicted_tree, gold_tree):

    predicted_edges = list(predicted_tree.edges())
    gold_edges = list(gold_tree.edges())
    tp = len([edge for edge in predicted_edges if gold_tree.has_edge(*edge)])
    fp = len([edge for edge in predicted_edges if not gold_tree.has_edge(*edge)])
    fn = len([edge for edge in gold_edges if not predicted_tree.has_edge(*edge)])
    if tp == 0:
        # print(f' tp {tp}, fp {fp}, fn {fn}')
        return 0, 0, 0
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (recall * precision) / (recall + precision)
    return precision, recall, f1




