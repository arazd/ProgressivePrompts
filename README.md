# Progressive Prompts

**Our work on Progressive Prompts is accepted to ICLR 2023!** 🎉

This repo includes an original implementation of Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis and Amjad Almahairi. ["Progressive Prompts: Continual Learning for Language Models"](https://arxiv.org/abs/2301.12314), ICLR 2023.

### Table of contents
* [Introduction](#star2-introduction)
* [What's in this repository](#question-whats-in-this-repository)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 
* [Contact](#raising_hand-questions)


## :star2: Introduction
We introduce **Progressive Prompts** – a novel Continual Learning (CL) approach for language models. Our
method is inspired by progressive networks ([A. Rusu et al., NeurIPS 2017](https://arxiv.org/pdf/1606.04671.pdf)), but is significantly more memory-efficient. In Progressive Prompts, we learn a separate set of virtual tokens, or ***soft prompt*** ([B. Lester et al., EMNLP 2021](https://arxiv.org/pdf/2104.08691.pdf)), for each incoming task and sequentially concatenate it with previously learned prompts. 

Our method can: 

1) **alleviate catastrophic forgetting**; since it preserves the knowledge acquired by previous prompts, and 
2) **transfer knowledge to future tasks**; since new prompts are sequentially concatenated with all prior prompts.

![Progressive Prompts schematics](/images/illustration.png)
Figure: *Illustrating our proposed method **Progressive Prompts** and contrasting it with a simple
adaptation of progressive networks using prompt tuning. In the simple adaptation of progressive
networks we learn a separate prompt and repeat the frozen input embeddings for each new task.
This setup requires repeating input tokens for each task. In Progressive Prompts we use the same
input and progressively append new prompt for each new task. Prior task prompts are not modified
by the addition of new prompts.*

## :question: What's in this repository

This is our code structure:

```
|_T5_codebase/
      |_t5_dataset.py --> T5 Dataset class for reading and processing datasets
      |_t5_continual.py --> Model class for T5 with prompt tuning and continual learning functions
      |_train_t5_cl.py --> Code to run continual learning experiments with T5
      
|_BERT_codebase/
      |_dataset_utils.py --> BERT Dataset class for reading and processing datasets
      |_model_utils.py --> Model class for BERT with prompt tuning and fine-tuning functions
      |_continual_learning_utils.py --> Continual Learner class for Progressive Prompts (with BERT)
      |_continual_learning_one_head.py --> Continual Learner class for regularization-based CL approaches for BERT 
      |_train_cl2.py --> Code to run continual learning experiments with BERT
      
|_datasets/src/data/ --> CL datasets from Zhang et. al., 2015
      |_amazon --> Amazon reviews (zip archive, since dataset is not available through HuggingFace datasets)
      (the rest of datasets can be either accessed through HuggingFace or downloaded by instructions below)
```

**Note**: we access most of the datasets for our experiments through HuggingFace datasets, including CL datasets from Zhang et. al., 2015. Since only one CL datasets from Zhang et. al. is not available on HuggingFace - Amazon Reviews, we uploaded its archived train / test data to ```datasets/src/data/amazon/```. To access the rest of CL datasets (Yelp, Yahoo, AG, DbPedia), you can either use their HuggingFace names in our training script or download them from [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq) to ```datasets/src/data/```.

## :wrench: Installation

Our implementation is based on PyTorch and HuggingFace (transformers + datasets). 

Requirements:
* Python 3.8.5
* Pytorch 1.10.0
* transformers 4.20.0
* datasets 2.3.2
* tqdm, sklearn, numpy, pandas

Step-by-step instructions to get you running Progressive Prompts:

### 1) Clone this repository to your local machine:

```bash
git clone https://github.com/arazd/ProgressivePrompts    
```  

A folder called ```ProgressivePrompts``` with all the codebase should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://docs.conda.io/en/latest/miniconda.html).

To run Progressive Prompts code on GPU, make sure that you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. In our implementation, we used and CUDA 11.0.

You can re-create our conda enviroment from ```environment.yaml``` file:

```bash
cd ProgressivePrompts
conda env create -f environment.yaml
```

Your conda should start downloading and extracting packages. This can take ~15-20 minutes.

### 3) Activate the environment:

Your environment should be called ```nlp```, and you can activate it now to run the scripts:

```bash
conda activate nlp
```

## :zap: How to run 

For example, to run Progressive Prompts with T5-large on four tasks (IMDb, CB, SST-2 and DbPedia):
```bash
cd T5_codebase

python train_t5_cl.py --task_list imdb cb sst2 dbpedia_14 --select_k_per_class 1000 \
--lr 0.3 --num_epochs 10 --freeze_weights 1 --prefix_len 10 \
--model_name t5-large --early_stopping 1 \
--save_name T5_experiment --save_dir my_path_to_save_directory
```

In the example above, we froze weights and trained a prompt of size 10 (per task) for 10 epochs. We also limited data to 1000 samples per class. 
For other arguments and their descriptions, please check ```T5_codebase/train_t5_cl.py``` file.


To train Progressive Prompts on the same four tasks with BERT-base:
```bash
cd BERT_codebase

python train_cl2.py --task_list imdb cb sst2 dbpedia_14  --select_k_per_class 1000 \
--lr 3e-5 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings \
--prompt_tuning 1 --prefix_len 10 --seq_len 450 --one_head 0 \
--model_name bert-base-uncased --early_stopping 1 \
--save_name BERT_experiment --save_dir my_path_to_save_directory
```

Note how soft prompts for BERT need to be trained with smaller learning rate and higher number of epochs. 
We also have some other BERT-specific arguments, one_head controls whether to use a separate head for each task, freeze_except allows to freeze all weights except word embeddings (since we include prompt tokens into vocabulary for BERT implementation), seq_len controls max input length (without prompt), prompt_tuning flag signals if we are doing prompt tuning.
For other arguments and their descriptions, please check ```BERT_codebase/train_cl2.py``` file.

<!--
The configuration keys are as follows:
| Argument |   Default     |  Description |
|----------|:-------------:|------:   |
| col 1 is |  left-aligned | $1600    |
| col 2 is |    centered   |   $12.   |
| col 3 is | right-aligned |    $1    |
-->


## :raising_hand: Questions
If you have any questions about the paper or code, please contact Anastasia Razdaibiedina (anastasia.razdaibiedina[at]mail.utoronto.ca) or open an issue. 

## :books: Citation
If you use our code in your research, please cite our work:
```bibtex
@inproceedings{razdaibiedina2023progressive,
   title={Progressive Prompts: Continual Learning for Language Models},
   author={Razdaibiedina, Anastasia and Mao, Yuning and Hou, Rui and Khabsa, Madian and Lewis, Mike and Almahairi, Amjad},
   booktitle={International Conference on Learning Representations},
   year={2023}
}
```

<!--
@article{razdaibiedina2023progressive,
  title={Progressive Prompts: Continual Learning for Language Models},
  author={Razdaibiedina, Anastasia and Mao, Yuning and Hou, Rui and Khabsa, Madian and Lewis, Mike and Almahairi, Amjad},
  journal={arXiv preprint arXiv:2301.12314},
  year={2023}
}
-->
