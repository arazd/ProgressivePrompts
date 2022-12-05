import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import dataset_utils, model_utils, continual_learning_utils, continual_learning_one_head



def main(args):

    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_file = os.path.join(save_path, 'LR_logs')

    if args.one_head == 0:
        prefix_MLP = args.prefix_MLP
        #prefix_MLP = None
        CL_class = continual_learning_utils.ContinualLearner(args.model_name,
                                                             args.task_list,
                                                             batch_size=args.batch_size,
                                                             select_k_per_class=args.select_k_per_class,
                                                             memory_perc=args.memory_perc,
                                                             #block_attn=0,
                                                             freeze_weights=args.freeze_weights,
                                                             freeze_except=args.freeze_except,
                                                             lr=args.lr,
                                                             seq_len=args.seq_len,
                                                             cls_idx_override=args.cls_idx_override,
                                                             early_stopping= args.early_stopping==1,
                                                             prefix_len=args.prefix_len,
                                                             prefix_MLP=prefix_MLP,
                                                             bottleneck_size=args.bottleneck_size,
                                                             do_repeats= args.do_repeats==1,
                                                             same_prompt=args.same_prompt==1,
                                                            )


        if args.multitask == 1:
            print("Multi-task learning")
            results_dict = CL_class.multi_task_training(num_epochs=args.num_epochs)
        else:
            results_dict = CL_class.continual_training(num_epochs=args.num_epochs,
                                                       data_replay_freq=args.data_replay_freq,
                                                       prompt_tuning=args.prompt_tuning==1,
                                                       prompt_init=args.prompt_init,
                                                       save_prompt_path='None' if args.save_prompt==0 else save_path,
                                                       save_results_path=save_path,
                                                       )
        np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)


    ## IDBR results
    else:
        CL_class = continual_learning_one_head.ContinualLearnerIDBR(args.model_name,
                                                                    args.task_list,
                                                                    batch_size=args.batch_size,
                                                                    select_k_per_class=args.select_k_per_class,
                                                                    memory_perc=args.memory_perc,
                                                                    #block_attn=0,
                                                                    freeze_weights=0,
                                                                    freeze_except=args.freeze_except,
                                                                    lr=args.lr,
                                                                    seq_len=args.seq_len,
                                                                    cls_idx_override=args.cls_idx_override,
                                                                    early_stopping= args.early_stopping==1,
                                                                    )
        if args.multitask == 1:
            print("Multi-task learning IDBR style")
            results_dict = CL_class.multi_task_training(num_epochs=args.num_epochs)
        else:
            print("Training in IDBR style")
            results_dict = CL_class.continual_training(num_epochs=args.num_epochs,
                                                       data_replay_freq=args.data_replay_freq,
                                                       regspe=args.regspe,
                                                       reggen=args.reggen,
                                                       tskcoe=args.tskcoe,
                                                       nspcoe=args.nspcoe,
                                                       disen=args.disen==1)

        np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features (should not be changed)',
        default='/data/home/arazdai/CL/all_CL_results/'
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
        '--memory_perc',
        type=float,
        help='Percentage of examples from previous tasks to use for data replay; if 0 then no data replay',
        default=0 #0.01
    )

    parser.add_argument(
        '--data_replay_freq',
        type=int,
        help='Data replay happens after every X samples (if -1 then no data replay)',
        default=-1 #9
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
        default=512
    )


    parser.add_argument(
        '--prefix_len',
        type=int,
        help='Soft prompt length (same for each task)',
        default=10
    )

    parser.add_argument(
        '--prefix_MLP',
        type=str,
        help='Whether to use MLP reparametrization on prompt embeddings (default = "None", no reparametrization)',
        default='None'
    )

    parser.add_argument(
        '--bottleneck_size',
        type=int,
        help='Bottleneck size in case of MLP reparametrization',
        default=800
    )


    parser.add_argument(
        '--num_repeats',
        type=int,
        help='Number of sentence repeats after the original sentence',
        default=0
    )

    parser.add_argument(
        '--block_attn',
        type=int,
        help='Whether to use blockwise causal attention',
        default=0
    )

    parser.add_argument(
        '--select_k_per_class',
        type=int,
        help='Select k examples from each class (default -1, i.e. no changes to the original dataset)',
        default=-1
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=3e-5
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
        '--prompt_tuning',
        type=int,
        help='Perform prompt tuning (1 - True, 0 - False)',
        default=0
    )

    parser.add_argument(
        '--same_prompt',
        type=int,
        help='Whether to use the same prompt for all tasks (1 - True, 0 - False)',
        default=0
    )

    parser.add_argument(
        '--save_prompt',
        type=int,
        help='Save prompts in np arrays after training (1 - True, 0 - False)',
        default=0
    )

    parser.add_argument(
        '--prompt_init',
        type=str,
        help='Initialization of next task prompts, if None - init from random word emb in the vocabulary',
        default='None'
    )


    parser.add_argument(
        '--do_repeats',
        type=int,
        help='Perform progressive prompt tuning with repeated input (1 - True, 0 - False)',
        default=0
    )


    parser.add_argument(
        '--cls_idx_override',
        type=int,
        help='Position of classification token; by default will use current task k token - CLSk',
        default=None
    )

    parser.add_argument(
        '--multitask',
        type=int,
        help='Perform multi-task learning (1 - True, 0 - False)',
        default=0
    )

    parser.add_argument(
        '--early_stopping',
        type=int,
        help='Perform early_stopping to select model at each task (1 - True, 0 - False)',
        default=1
    )

    ##### IDBR style utils #####
    parser.add_argument(
        '--one_head',
        type=int,
        help='Perform training with one large head for all tasks like in IDBR paper (1 - True, 0 - False)',
        default=0
    )

    parser.add_argument(
        '--regspe',
        type=float,
        help='Regularization coef for task-specific features',
        default=0.5
    )

    parser.add_argument(
        '--reggen',
        type=float,
        help='Regularization coef for general features',
        default=0.5
    )

    parser.add_argument(
        '--tskcoe',
        type=float,
        help='Task loss coef',
        default=0.0
    )

    parser.add_argument(
        '--nspcoe',
        type=float,
        help='NSP loss coef',
        default=0.0
    )

    parser.add_argument(
        '--disen',
        type=int,
        help='Perform IDBR training (disentanglement)',
        default=0
    )

    main(parser.parse_args())
