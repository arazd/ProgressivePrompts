import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import dataset_utils, model_utils


task_to_num_labels = {
    'cola': 2,
    'sst2': 2,
    'yelp_review_full': 5,
    'ag_news': 4,
    'yahoo_answers_topics': 10
}
glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']


def train_on_task(trainer,
                  task,
                  num_epochs,
                  batch_size,
                  k=-1,
                  cls_idx=0,
                  ):
    print('Starting task ', task)
    du = dataset_utils.Dataset(task=task, tokenizer=trainer.tokenizer)
    data_params = {'repeats': trainer.num_repeats,
                   'batch_size': batch_size,
                   'max_length': trainer.max_length,
                   'special_tokens_list': trainer.special_tokens_list,
                   #'select_k_per_class': k
                   }
    benchmark = 'glue' if task in glue_datasets else None
    val_split = 'validation' if task in glue_datasets else 'test'
    dataloader_train = du.get_dataset(benchmark=benchmark, split='train', select_k_per_class=k, **data_params)
    k_val = -1 if k==-1 else int(k*0.1)
    dataloader_val = du.get_dataset(benchmark=benchmark, split=val_split, select_k_per_class=k_val, **data_params)

    print("Trainig set size = ", len(dataloader_train)*batch_size)
    scores_list = trainer.train(dataloader_train,
                                task,
                                epochs=num_epochs,
                                cls_idx=cls_idx,
                                #cls_idx=trainer.max_length, # CHANGE THIS CLS IDX
                                dataloader_val=dataloader_val)

    return scores_list



def continual_training(trainer,
                       tasks=[],
                       num_epochs=5,
                       batch_size=8,
                       k=-1):
    results_dict = {}
    for i, task in enumerate(tasks):
        print('TASK ', task)
        val_scores = train_on_task(trainer, task, num_epochs, batch_size, k=k)
        results_dict[task] = val_scores
    return results_dict



def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_file = os.path.join(save_path, 'LR_logs')
    logging.basicConfig(filename=log_file,level=logging.DEBUG)

    logging.info("starting training script")

    task_list = args.task_list # ['cola']
    special_tokens_list = ["[CLS"+str(i+1)+"]" for i in range(len(task_list))] #[ "[CLS1]" ],
    num_labels = [task_to_num_labels[t] for t in task_list] # CHANGE THIS !!!!

    model_name = args.model_name #"bert-base-uncased"
    trainer = model_utils.ModelForCL( model_name,
                                      tasks=task_list,
                                      num_labels=num_labels,
                                      blockwise_causal_attention= (args.block_attn==1),
                                      special_tokens_list=special_tokens_list,
                                      init_token="[CLS]",
                                      freeze_weights= (args.freeze_weights==1),
                                      freeze_except=args.freeze_except, #by default 'word_embeddings',
                                      lr=args.lr,
                                      num_repeats=args.num_repeats,
                                      max_length=args.seq_len, # max sequence length in #tokens
                                      cls_idx_override=args.cls_idx_override,
                                      same_prompt=args.same_prompt==1,
                                      )

    # task = trainer.current_task
    # du = dataset_utils.Dataset(task=task, tokenizer=trainer.tokenizer)
    # data_params = {'repeats': trainer.num_repeats,
    #                'batch_size': args.batch_size,
    #                'max_length': trainer.max_length,
    #                'special_tokens_list': trainer.special_tokens_list}
    # dataloader_train = du.get_dataset(benchmark='glue', split='train', **data_params)
    # dataloader_val = du.get_dataset(benchmark='glue', split='validation', **data_params)
    #
    #
    # scores_list = trainer.train(dataloader_train,
    #                             task,
    #                             epochs=args.num_epochs,
    #                             cls_idx=trainer.max_length,
    #                             dataloader_val=dataloader_val)
    #
    # if not os.path.exists( os.path.join(save_path, 'val_scores.npy') ):
    #     np.save(os.path.join(save_path, 'val_scores.npy'), scores_list)
    #
    # else:
    #     np.save(os.path.join(save_path, 'val_scores2.npy'), scores_list)

    results_dict = continual_training( trainer,
                                       tasks=task_list,
                                       num_epochs=args.num_epochs,
                                       batch_size=args.batch_size,
                                       k=args.select_k_per_class)
    np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features (should not be changed)',
        default='/data/home/arazdai/CL' #'/scratch/hdd001/home/anastasia/CL/'
    )

    parser.add_argument(
        '--save_name',
        type=str,
        help='folder name to save',
        required=True
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the model used for training',
        default="bert-base-uncased"
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs to train model',
        default=5
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=8
    )

    parser.add_argument(
        '--seq_len',
        type=int,
        help='Length of a single repeat (in #tokens)',
        default=150
    )

    parser.add_argument(
        '--num_repeats',
        type=int,
        help='Number of sentence repeats',
        required=True
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=2e-5
    )

    parser.add_argument(
        '--block_attn',
        type=int,
        help='Whether to use blockwise causal attention',
        default=0
    )

    parser.add_argument(
        '--same_prompt',
        type=int,
        help='Whether to use the same prompt across all tasks',
        default=0
    )

    parser.add_argument(
        '--select_k_per_class',
        type=int,
        help='Select k examples from each class (default -1, i.e. no changes to the original dataset)',
        default=-1
    )

    parser.add_argument(
        '--freeze_weights',
        type=int,
        help='Whether to freeze model weigts (except word emb)',
        default=0
    )

    parser.add_argument(
        '--freeze_except',
        type=str,
        help='If freeze_weights==1, freeze all weights except those that contain this keyword',
        default='word_embeddings'
    )

    parser.add_argument(
        '--task_list',
        nargs='+',
        help='List of tasks for training',
        required=True
    )

    parser.add_argument(
        '--cls_idx_override',
        type=int,
        help='Position of classification token; by default will use current task k token - CLSk',
        default=None
    )

    main(parser.parse_args())
