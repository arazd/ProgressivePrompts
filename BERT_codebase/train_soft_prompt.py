import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import dataset_utils, model_utils, continual_learning_utils, continual_learning_one_head


def get_prefix_net(bottleneck_size = 800, network_type='MLP-1'):

    if network_type == 'MLP1':
        prefix_MLP = nn.Sequential(
            nn.Linear(768, bottleneck_size),
            nn.ReLU(),
            #nn.Linear(bottleneck_size, bottleneck_size),
            #nn.Tanh(),
            nn.Linear(bottleneck_size, 768),
        )

    elif network_type == 'MLP2':
        prefix_MLP = nn.Sequential(
            nn.Linear(768, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.Tanh(),
            nn.Linear(bottleneck_size, 768),
        )

    elif network_type == 'transformer':
        device = 'cuda'
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2, dropout=0.05).to(device)
        prefix_MLP = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

    return prefix_MLP


def train_prompt(args, CL_class, epochs = 50, cls_idx = 0, prompt_tuning=True):
    val_scores = []
    task = args.task
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataloader_train, dataloader_val = CL_class.tasks_data_dict[task]['train'], CL_class.tasks_data_dict[task]['val']
    #task = args.task #'cola'

    for epoch in range(epochs):
        print(epoch)
        CL_class.trainer.model.train().to(device)
        if args.prefix_len>0 and args.prefix_MLP != 'None':
            CL_class.trainer.prefix_MLP.train().to(device)

        for i, batch in enumerate(tqdm(dataloader_train)):
            batch = {k: v.to(device) for k, v in batch.items()}
            if prompt_tuning: # pass batch with prompt tuning
                CL_class.trainer.pass_batch_with_prompt(batch, task, device, prefix_len=args.prefix_len)
            else:  # regular pass batch
                CL_class.trainer.pass_batch(batch, task, device, cls_idx=0)

        CL_class.trainer.model.eval()
        if args.prefix_len>0 and args.prefix_MLP != 'None':
            CL_class.trainer.prefix_MLP.eval()

        if prompt_tuning:
            result = CL_class.trainer.eval_with_prompt(dataloader_val, task, None, cls_idx=0)
        else:
            result = CL_class.trainer.eval(dataloader_val, task, None, cls_idx=0)
        print(' result = ',result, '\n')
        val_scores.append(result)
    return val_scores


def main(args):

    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_file = os.path.join(save_path, 'LR_logs')

    if args.prefix_MLP == 'None':
        prefix_net = None
    else:
        prefix_net = get_prefix_net(bottleneck_size = args.bottleneck, network_type=args.prefix_MLP)

    CL_class= continual_learning_utils.ContinualLearner(args.model_name,
                                                        [args.task], # one task as a list of tasks
                                                        batch_size=args.batch_size,
                                                        select_k_per_class=args.select_k_per_class,
                                                        memory_perc=args.memory_perc,
                                                        block_attn=0,
                                                        freeze_weights=args.freeze_weights,
                                                        freeze_except=args.freeze_except,
                                                        lr=args.lr,
                                                        seq_len=args.seq_len,
                                                        cls_idx_override=args.cls_idx_override,
                                                        early_stopping= args.early_stopping==1,
                                                        prefix_MLP=prefix_net,
                                                        prefix_len=args.prefix_len,
                                                        )

    results = train_prompt(args, CL_class, epochs = args.num_epochs, cls_idx = 0, prompt_tuning= args.prompt_tuning==1)
    np.save(os.path.join(save_path, 'results.npy'), results)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features (should not be changed)',
        default='/data/home/arazdai/CL/'
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
        '--prompt_tuning',
        type=int,
        help='If 1, use prompt_batch_pass / eval (takes into account prefix MLP and wte), else use normal batch pass',
        default=1
    )

    parser.add_argument(
        '--prefix_len',
        type=int,
        help='Length of the soft prompt to be tuned (default 0 - no soft prompt)',
        default=0
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

    # parser.add_argument(
    #     '--task_list',
    #     nargs='+',
    #     help='List of tasks for training',
    #     required=True
    # )

    parser.add_argument(
        '--task',
        type=str,
        help='Task to train on (only one)',
        required=True
    )

    parser.add_argument(
        '--prefix_MLP',
        type=str,
        help='Type of network to pass soft prompt through',
        default='None'
    )

    parser.add_argument(
        '--bottleneck',
        type=int,
        help='Bottleneck size of prefix MLP',
        default=800
    )

    parser.add_argument(
        '--cls_idx_override',
        type=int,
        help='Position of classification token; by default will use current task k token - CLSk',
        default=None
    )

    # parser.add_argument(
    #     '--multitask',
    #     type=int,
    #     help='Perform multi-task learning (1 - True, 0 - False)',
    #     default=0
    # )

    parser.add_argument(
        '--early_stopping',
        type=int,
        help='Perform early_stopping to select model at each task (1 - True, 0 - False)',
        default=0
    )

    main(parser.parse_args())
