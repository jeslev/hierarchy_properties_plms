# Train

- hierarchy_props_v0: Using ctp wordnet definitiosn alignment
- hierarchy_props_v1: Using bansal wordnet definitions directly
- hierarchy_props: Similar to V1 but generating soft positive examples

These dataset folders are considered different because we run subsamples to normalize 
the amount of triplets for each entity. 

Train using 2 methods
1. Dev on triplets
- We use triplets only with positive triplets (tripletavg_v2_pos.db). This can be considered as hard examples 
since they are very close in the taxonomy. This corresponds using the data folder hierarchy_props_v1 
- When training with bert-base-cased, optuna finds the best hyperparameters with values
- Dev score: 0.68, batch= 8-16-32, epochs=3,4 ,lr= 1.0193555926254115e-07, 1.1993535601877016e-07
 1.1956809968455346e-07 warmup 0.1 0.05. (similar results for tripletavg.db using hierarchy_props_vo)
- We use sof positives examples to add more comparisons to a given pair (5 by pair). We obtain similar results
than before (tripletavg_soft_pos.db)
- We now generate up to 30 soft positive triplets per example. (0.6954) with batch 8, epochs 3, lr.1.78e-07 warmup 0.05 tripletavg_soft_pos_x30.db
- Apparently, I was not using the soft negatives until now. -.-. Rerunning
- We train on all properties, using mean pooling for training, dev avg tokens
- We train only on property 1 (dev prop=1) using avg tokens (all)
- We train only on property 1 (dev prop=1) using cls token
2.Dev on taxonomy reconstruction

- We use triplets only with positive triplets (taxonomy_pos_v2.db). This can be considered as hard examples 
since they are very close in the taxonomy. This corresponds using the data folder hierarchy_props_v1 
- When training with bert-base-cased, optuna finds the best hyperparameters with values
  - Dev score: 0.37 atch: 32, 16 epochs 2, lr=0.00947214618315449 lr = 0.0065828777025057
  warmup=0.15 0.05 (similar results for taxonomy.db using hierarchy_props_vo)
- We use sof positives examples to add more comparisons to a given pair (5 by pair). We obtain similar results
than before (taxonomy_soft_pos.db)
- similar adding more soft positives. We obtained similar results than before (0.369) (taxonomy_soft_pos_x30.db)
- Now testing roberta large and other hyperparams (smaller range lr, epochs) (taxonomy_soft_pos_x30_v2.db)
- Apparently, I was not using the soft negatives until now. -.-. Rerunning
- We train on all properties, using mean pooling for training, dev avg tokens
- We train only on property 1 (dev prop=1) using avg tokens (all)
-  We train only on property 1 (dev prop=1) using cls token

## 08/02/2023
After correction on the computing metrics for taxonomy and computing zero-shot scores of raw models, 
and geneating new properties triplets to train. We rerun the training with 3 approaches

### Training with train_hard_soft_neg_sample_5x_p123689.json
 all properties (except 5)
1. With triplets x5 negatives - evaluation triplets
   1. CLS (train_triplet_x5_pall_cls.db) (done)
   2. AVG (train_triplet_x5_pall_avg.db) (done)
      - 60.78 2 epochs
2. With triplets x5 negatives - evaluation taxonomy
   1. CLS (train_taxo_x5_pall_cls.db) (done)
   2. AVG (train_taxo_x5_pall_avg.db) (done)
      - 4 epochs. 36.8 f1
      - Rerun fixing DiGraph error (maybe it improves?)
      	NO!. With 18 runs we get: 36.6 epochs 3,5,9 , lr:0.027, 0.003,0.0012
   
#### With bert-base
1. Evaluation 
	1. AVG - 0.60 params:  epochs: 2,3 batch:64, lr:1.001e-7, 1.073e-7, 2.259e-7
	2. CLS - 0.595 params, epochs:2,3 batch 32, lr: 1.01e-07, lr: 1.003e-7
2. Taxonomy
	1. AVG - 0.37 params: epochs:3-4, batch:32,64 lr:0.042,0.008,0.01
	2. CLS - 0.362, epochs=2,5 batch: 64,32, lr = 0.0013, 0.003, 0.24 (epoch5)
	
#### With roberta-base
1. Evaluation 
	1. AVG - 
	2. CLS -  0.576 (epochs 3,2 lr 3.4e07, 2.5e-6 (done osirim, rob_train_triplet_x5_pall_cls.db)
2. Taxonomy
	1. AVG - 
	2. CLS - 0.3677 epochs 6, lr 0.09, 0.01 (done osirim, rob_train_taxo_x5_pall_cls.db)



#### With Distil-RoB
1. Evaluation 
	1. AVG - 
	2. CLS - 0.5997, epochs2, lr: 1.9e-7,2.1e-7 (done - JZ, train_triplet_x5_pall_cls_distilrob.db)
2. Taxonomy
	1. AVG - 
	2. CLS - 0.36517 epochs: 3-4, lr 0.0699 (done - JZ, train_taxo_x5_pall_cls_distilrob.db)

#### With MPNET
1. Evaluation 
	1. AVG - 
	2. CLS - 0.60 epochs:2 lr: 3e-7. re-running - JZ, train_triplet_x5_pall_cls_mpnet.db)
2. Taxonomy
	1. AVG - 
	2. CLS - (canceled - JZ, train_taxo_x5_pall_cls_mpnet.db)


From the previous results we can choose a model to analyze (bert or roberta) with the large version and single properties + sbert (distilrob if roberta, mpnet if bert)


#### Models to train triplets, evaluting on properties, each epoch print all properties score
- BERT-BASE (running jz)
CLS: epochs 5, batch 32, lr 1e-7, 2e-7
- Rob-base
CLS: epochs: epochs 5, batch: 32, lr: 1e-7, 2e-7
- Distil-Rob
CLS: epochs: 4, batch 32, lr: 1e-7, 2e-7, 3e-7
- MPNet
CLS: epochs: 4, batch 32, lr: 1e-7, 2e-7, 3e-7

Total:
3lrx4models = 12 models

We evaluate each model (with and without soft negativeS) using all properties (running JZ) with 6 different configurations (different lrs (1e-7, 2e-7, 3e-7 and cls or pool)

then, we analyze the results and we select one of these model to train per single property (with or without soft negatives)

- From results:
Training with only hard positive gives better performance
We seleted BERT-L, DistilRob, and MPNet to have one representant (with the best performance) for each model

We tried optuna exploration with soft+hard negatives, and even in taht situation, only hard postivie gives better results.

### Train with each property

We evaluate now the impact of each property on the other properties
Configuration 
 BERT-L (cls,2e-7)3, DistilRob (mean, 1e-7), MPNet (mean, 1-2e-7)
 6 properties x 3 models (4 models mpnet trying 2 lrs) = 24 models

(done in jz)
Evaluation with all combinations are made (in hpatterns_1p folder). Results are extracted in a
Google sheet, for the 3 models. The configuration used are in the last table in paper.
