import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

from t5_continual import T5ContinualLearner


def main(args):
    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    task_list = args.task_list

    model_name = args.model_name
    continual_learner = T5ContinualLearner(model_name,
                                           task_list,
                                           batch_size=args.batch_size,
                                           select_k_per_class=args.select_k_per_class,
                                           prefix_len=args.prefix_len,
                                           freeze_weights=args.freeze_weights==1,
                                           freeze_except=args.freeze_except,
                                           lr=args.lr,
                                           seq_len=args.seq_len,
                                           early_stopping=args.early_stopping==1,
                                           prefix_MLP=args.prefix_MLP,
                                           prefix_path=args.prefix_path if args.prefix_path!='' else None,
                                           mlp_layer_norm=args.mlp_layer_norm==1,
                                           bottleneck_size=args.bottleneck_size,
                                           get_test_subset=args.get_test_subset==1,
                                           memory_perc=args.memory_perc
                                           )
    if args.get_test_subset==0:
        print("Not creating test subset")

    if args.multitask == 1:
        print('Multi task learning')
        results_dict = continual_learner.multi_task_training(num_epochs=args.num_epochs, save_path=save_path)
        np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)

    else:
        if args.num_epochs<=50:
            eval_every_N = 1
        elif args.num_epochs>50 and args.num_epochs<=200:
            eval_every_N = 5
        elif args.num_epochs>200:
            eval_every_N = 10

        results_dict = continual_learner.train_continual(continual_learner.task_list,
                                                        epochs=args.num_epochs,
                                                        save_path=save_path,
                                                        progressive=args.progressive==1,
                                                        eval_every_N=eval_every_N,
                                                        test_eval_after_every_task=args.test_eval_after_every_task==1,
                                                        data_replay_freq=args.data_replay_freq,
                                                        )
        np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
        np.save(os.path.join(save_path, 'prompts.npy'), continual_learner.previous_prompts.detach().cpu().numpy())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features (should not be changed)',
        default='/data/home/arazdai/T5_prompts/T5_continual/' #'/scratch/hdd001/home/anastasia/CL/'
    )

    parser.add_argument(
        '--save_name',
        type=str,
        help='folder name to save',
        required=True
    )

    parser.add_argument(
        '--task_list',
        nargs='+',
        help='List of tasks for training',
        required=True
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the model used for training',
        default="t5-base"
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs to train model',
        default=5
    )

    parser.add_argument(
        '--multitask',
        type=int,
        help='Whether to perform multi-task training',
        default=0
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
        help='Length of prompt (in #tokens)',
        default=10
    )

    parser.add_argument(
        '--prefix_path',
        type=str,
        help='path to a pre-trained progressive prefix (for superGLUE experiments)',
        default=''
    )


    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=0.3
    )


    parser.add_argument(
        '--memory_perc',
        type=float,
        help='Memory perc',
        default=0.01
    )

    parser.add_argument(
        '--data_replay_freq',
        type=float,
        help='Replay data every X iterations',
        default=-1
    )

    parser.add_argument(
        '--select_k_per_class',
        type=int,
        help='Select k examples from each class (default -1, i.e. no changes to the original dataset)',
        default=-1
    )

    parser.add_argument(
        '--test_eval_after_every_task',
        type=int,
        help='Whether to re-evaluate test accuracy after every task (0 - False, 1 - True)',
        default=0
    )

    parser.add_argument(
        '--progressive',
        type=int,
        help='Whether to concatenate prompts in a progressive way (0 - False, 1 - True)',
        default=1
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
        default='xxxxxxx' # freeze all
    )

    parser.add_argument(
        '--get_test_subset',
        type=int,
        help='Whether to create a separate test split',
        default=1
    )

    parser.add_argument(
        '--early_stopping',
        type=int,
        help='If early_stopping==1, do early stopping based on val accuracy',
        default=1 # freeze all
    )

    parser.add_argument(
        '--prefix_MLP',
        type=str,
        help='Type of MLP reparametrization (if None - use Lester original implementation)',
        default='None' # freeze all
    )

    parser.add_argument(
        '--mlp_layer_norm',
        type=int,
        help='Do layer norm in MLP',
        default=1 # use layer norm
    )

    parser.add_argument(
        '--bottleneck_size',
        type=int,
        help='MLP bottleneck size',
        default=800
    )

    main(parser.parse_args())
