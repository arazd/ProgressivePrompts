import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse
import matplotlib.pyplot as plt

import dataset_utils, model_utils
from itertools import cycle
from copy import deepcopy


class ResMLP(torch.nn.Module):
    def __init__(self, bottleneck_size, module_type='MLP1'):
        super().__init__()
        if module_type=='MLP1':
            self.module = nn.Sequential(
                nn.Linear(768, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, 768),
            )

        elif module_type=='MLP2':
            self.module = nn.Sequential(
                nn.Linear(768, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size),
                nn.Tanh(),
                nn.Linear(bottleneck_size, 768),
            )

        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

    def forward(self, inputs):
        return self.module(inputs) + inputs


def get_prefix_net(bottleneck_size = 800, network_type='MLP1'):

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

    elif 'residual' in network_type:
        prefix_MLP = ResMLP(bottleneck_size, module_type=network_type.split('_')[1])

    return prefix_MLP



class ContinualLearner:
    def __init__(self,
                 model_name,
                 task_list,
                 batch_size=8,
                 select_k_per_class=-1,
                 memory_perc=0,
                 #block_attn=0,
                 prefix_len=0,
                 freeze_weights=0,
                 freeze_except='word_embeddings',
                 lr=2e-5,
                 seq_len=512,
                 cls_idx_override=None,
                 early_stopping=True,
                 prefix_MLP='None',
                 do_repeats=False, # default setting is without repeats
                 bottleneck_size=800, # bottleneck size in case of using MLP reparametrization
                 same_prompt=False,
                 ):

        self.task_to_num_labels = {
            'cola': 2,
            'rte': 2,
            'mrpc': 2,
            'qqp': 2,
            'sst2': 2,
            'qnli': 2,
            'mnli': 3,

            'scicite': 3,
            'imdb': 2,

            'cb': 3,
            'copa': 2,
            'wic': 2,
            'boolq': 2,
            'multirc': 2,

            'yelp_review_full': 5,
            'ag_news': 4,
            'yahoo_answers_topics': 10,
            'dbpedia_14': 14,
            'amazon': 5,

            'yelp': 5,
            'ag': 4,
            'yahoo': 10,
            'proc_yahoo': 10,
            'dbpedia': 14,
        }
        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue = ['cb', 'copa', 'wic', 'wsc', 'boolq', 'record', 'multirc']

        self.task_list = task_list
        #self.special_tokens_list = ["[CLS"+str(i+1)+"]" for i in range(len(self.task_list))] #[ "[CLS1]" ],
        #self.special_tokens_list = []
        self.num_labels = [self.task_to_num_labels[t] for t in self.task_list]
        self.freeze_weights = freeze_weights
        self.prefix_len = prefix_len
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cls_idx_override = cls_idx_override
        self.same_prompt = same_prompt
        self.select_k_per_class = select_k_per_class
        self.memory_perc = memory_perc
        self.freeze_except = freeze_except
        self.early_stopping = early_stopping
        self.do_repeats = do_repeats

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_name = model_name #"bert-base-uncased"
        num_repeats = len(self.task_list)-1 if self.do_repeats else 0
        print('Max repeats = ', num_repeats)

        if prefix_MLP == 'None':
            prefix_MLPs = None
        else:
            print('Using MLP reparametrization with bottleneck = ', bottleneck_size)
            prefix_MLPs = {t: get_prefix_net(bottleneck_size = bottleneck_size, network_type=prefix_MLP) for t in self.task_list}

        self.trainer = model_utils.ModelForCL(self.model_name,
                                              tasks=self.task_list,
                                              num_labels=self.num_labels,
                                              #blockwise_causal_attention= (block_attn==1),
                                              prefix_len=self.prefix_len,
                                              freeze_weights= (self.freeze_weights==1),
                                              freeze_except=self.freeze_except,
                                              lr=self.lr,
                                              num_repeats=num_repeats, # default 0
                                              max_length=self.seq_len, # max sequence length in #tokens
                                              cls_idx_override=self.cls_idx_override,
                                              prefix_MLPs=prefix_MLPs,
                                              same_prompt=self.same_prompt,
                                              )
        self.trainer.model.to(self.device) # model to cuda
        if prefix_MLPs!=None:
            for t in self.task_list:
                self.trainer.prefix_MLPs[t].to(self.device)
        if self.early_stopping:
            self.best_model = deepcopy(self.trainer.model.state_dict()) # saving best model
            self.best_acc = 0.0 # best avg accuracy on seen tasks
        self.tokenizer = self.trainer.tokenizer
        self.tasks_data_dict = self.get_tasks_data_dict(self.select_k_per_class, memory_perc=self.memory_perc)


    def get_tasks_data_dict(self, k, memory_perc=0):
        # if k==-1: use all data, otherwise use k examples from class
        trainer = self.trainer
        tasks_data_dict = {}
        current_task_progressive_prompt = []

        k_val = -1 if k==-1 else max(int(k*0.15), 500)
        for task in self.task_list:
            tasks_data_dict[task] = {}
            print(task)
            if self.same_prompt: # same prompt for all tasks
                current_task_progressive_prompt = trainer.prefix_tokens_list[0]
            else:
                current_task_progressive_prompt += trainer.prefix_tokens_list[task]
            print(current_task_progressive_prompt)
            du = dataset_utils.Dataset(task=task, tokenizer=trainer.tokenizer)
            if self.same_prompt:
                tid = 0
            else:
                tid = self.task_list.index(task) # task id
            data_params = {'repeats': 0 if not self.do_repeats else tid, #trainer.num_repeats,
                           'batch_size': self.batch_size,
                           'max_length': trainer.max_length,
                           'prefix_tokens_list': current_task_progressive_prompt,
                           'prefix_len': self.prefix_len * (tid+1), # prompt * task_quantity
                           'do_repeats': self.do_repeats, # only applies to the first task in repeated set-up (if we format it according to repeats)
                           }
            benchmark = 'glue' if task in self.glue_datasets else 'super_glue' if task in self.superglue else None
            val_split = 'validation' if (task in self.glue_datasets or task in self.superglue) else 'test'
            dataloader_train = du.get_dataset(benchmark=benchmark, split='train', select_k_per_class=k, **data_params)
            if memory_perc>0:
                k_mem = max(int(len(dataloader_train)*memory_perc), 2)
                dataloader_mem = du.get_dataset(benchmark=benchmark, split='train',
                                                select_k_per_class=k_mem, **data_params)

            if k!=-1:
                if task in ['dbpedia', 'sst2']: k_val = int(k*0.1)
                else: k_val = int(k*0.15)
            dataloader_val, dataloader_test = du.get_dataset(benchmark=benchmark, split=val_split,
                                                             select_k_per_class=k_val, return_test_subset=True,
                                                             **data_params)

            tasks_data_dict[task]['train'] = dataloader_train
            if memory_perc>0: tasks_data_dict[task]['train_mem'] = dataloader_mem # for data replay
            tasks_data_dict[task]['val'] = dataloader_val
            tasks_data_dict[task]['test'] = dataloader_test

        return tasks_data_dict


    def change_attention_mask_for_tasks(self, task_num):
        #if self.do_repeats:
        # overriding attention mask for the next task
        print('updating attention mask for tasks #', task_num)
        num_repeats = task_num
        repeat_length = self.prefix_len + self.seq_len
        self.trainer.override_attention_mask(num_repeats, repeat_length, self.device)



    def eval_on_tasks(self, val_scores, split='val', prompt_tuning=True, original_task_id=None, tasks_to_eval=None):
        self.trainer.model.eval()
        if self.prefix_len>0 and self.trainer.prefix_MLPs != None:
            for task in list(self.tasks_data_dict): # put all MLPs into eval mode
                self.trainer.prefix_MLPs[task].eval()

        if tasks_to_eval==None: # if not specified, eval on all tasks
            tasks_to_eval = self.tasks_data_dict

        for task in list(tasks_to_eval):
            dataloader_val = self.tasks_data_dict[task][split]
            if prompt_tuning: # special eval for prompts (we use custom pos ids)
                if self.same_prompt:
                    tid=0
                else:
                    tid = self.task_list.index(task)
                if self.do_repeats:
                    self.change_attention_mask_for_tasks(tid) # change attention mask in case of "repeats" set up
                pos_id = self.get_pos_id(tid)
                cls_idx = self.seq_len + self.prefix_len*tid if not self.do_repeats \
                          else self.seq_len + (self.seq_len + self.prefix_len) * tid
                          #else (self.seq_len + self.prefix_len)*tid
                result = self.trainer.eval_with_prompt(dataloader_val, task, None,
                                                       cls_idx=cls_idx,
                                                       custom_pos_ids=True,
                                                       pos_ids=pos_id)
            else: # regular eval for fine-tuning
                cls_idx = 0
                result = self.trainer.eval(dataloader_val, task, None, cls_idx)
            print(task, ' result = ',result)
            val_scores[task].append(result)

        # restore original attention mask for the current task after evaluation
        if self.do_repeats and original_task_id!=None:
            self.change_attention_mask_for_tasks(original_task_id)
            print('restored attn mask for task ', original_task_id)

        return val_scores



    def update_best_model(self, curr_task, val_scores, tasks_to_eval=None):
        #idx = list(self.tasks_data_dict).index(curr_task) # look tasks up to curr task
        #seen_tasks = [t for t in list(self.tasks_data_dict)[:idx+1]]
        #avg_acc = np.mean([val_scores[task][-1] for task in seen_tasks])
        if tasks_to_eval==None:
            tasks_to_eval = self.task_list
        avg_acc = np.mean([val_scores[task][-1] for task in tasks_to_eval])
        # only update if we are starting a new task OR if acc gets better
        if avg_acc > self.best_acc:
            print('NEW BEST MODEL acc=', avg_acc)
            self.best_acc = avg_acc
            self.best_model = deepcopy(self.trainer.model.state_dict())


    # for prompt tuning set up we use custom position ids
    # Hello world :) [pad] [pad] ... [pad] [pre0_1] [pre0_2] [pre0_3] [pre1_1] [pre1_2] [pre1_3]
    # 1  2  3  4  5  6      7        400    0        401      402       0        403      404
    # def get_pos_id(self, task_id):
    #     s = self.seq_len+1
    #     pos_id = list(np.arange(1, s))
    #     for k in range(task_id+1):
    #         pos_id += [0] + list(np.arange(s, s + self.prefix_len-1))
    #         s = s + self.prefix_len - 1
    #     return torch.tensor(pos_id)

     # for prompt tuning set up we use custom position ids
    def get_pos_id(self, task_id):
        s = self.seq_len+1

        if not self.do_repeats:
        # (progressive) prompt tuning set-up
        # Hello world :) [pad] [pad] ... [pad] [pre0_1] [pre0_2] [pre0_3] [pre1_1] [pre1_2] [pre1_3]
        # 1  2  3  4  5  6      7        400    0        401      402       0        403      404
            pos_id = list(np.arange(1, s))
            for k in range(task_id+1):
                pos_id += [0] + list(np.arange(s, s + self.prefix_len-1))
                s = s + self.prefix_len - 1
            return torch.tensor(pos_id)

        else:
        # prompt tuning with repeats
        # Hello world :) [pad] [pad] ... [pad] [pre0_1] [pre0_2] [pre0_3] | Hello world :) [pad] [pad] ... [pad][pre1_1] [pre1_2] [pre1_3]
        # 1  2  3  4  5  6      7        400    0        401      402     | 1  2  3  4  5  6      7        400    0        401      402
            pos_id = list(np.arange(1, s)) + [0] + list(np.arange(s, s + self.prefix_len-1))
            pos_id *= task_id+1
            return torch.tensor(pos_id)



    def create_memory_replay_generators(self, task, split='train_mem'): # creating previous tasks memory buffers
        print('Creating generators for previous tasks ...')
        tasks_to_generators = {}
        curr_task_num = list(self.tasks_data_dict).index(task)
        for idx in np.arange(curr_task_num):
            prev_task = list(self.tasks_data_dict)[idx]
            print(prev_task)
            tasks_to_generators[prev_task] = iter(self.tasks_data_dict[prev_task][split])
        return tasks_to_generators


    def memory_replay(self, tasks_to_generators, cls_idx):
        # for each memory buffer in tasks_to_generators perform memory replay
        for prev_task in tasks_to_generators:
            generator_mem1 = tasks_to_generators[prev_task]
            try:
                # Samples the batch
                b = next(generator_mem1)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator_mem1 = iter(self.tasks_data_dict[prev_task]['train_mem'])
                tasks_to_generators[prev_task] = generator_mem1
                b = next(generator_mem1)

            b = {k: v.to(self.device) for k, v in b.items()}
            self.trainer.pass_batch(b, prev_task, self.device, cls_idx=cls_idx)


    def train_on_one_task(self,
                          task,
                          data_replay_freq = -1, # if -1 no data replay, else replay after N samples
                          prompt_tuning = True,
                          num_epochs = 5):
        self.best_acc = 0.0 # our baseline accuracy is 0
        val_scores = {x: [] for x in list(self.tasks_data_dict)}
        device = self.device

        if prompt_tuning:
            if self.same_prompt:
                task_id = 0
            else:
                task_id = self.task_list.index(task)
            pos_id = self.get_pos_id(task_id)
            cls_idx = self.seq_len + self.prefix_len*task_id if not self.do_repeats \
                      else self.seq_len + (self.seq_len + self.prefix_len) * task_id
        else:
            task_id = None # we do not need task id for eval in case of regular fine-tuning

        for epoch in range(num_epochs):
            print(epoch)
            self.trainer.model.train().to(device)
            if self.prefix_len>0 and self.trainer.prefix_MLPs != None:
                self.trainer.prefix_MLPs[task].train().to(device)

            if data_replay_freq != -1:
                tasks_to_generators = self.create_memory_replay_generators(task, split='train_mem')

            for i, batch in enumerate(tqdm(self.tasks_data_dict[task]['train'])):
                batch = {k: v.to(device) for k, v in batch.items()}

                if prompt_tuning: # tune only soft prompt
                    batch['position_ids'] = pos_id.to(self.device) # custom pos ids for prompts
                    self.trainer.pass_batch_with_prompt(batch, task, self.device,
                                                        prefix_len=self.prefix_len,
                                                        cls_idx=cls_idx,
                                                        custom_pos_ids=True)
                else: # regular fine-tuning
                    self.trainer.pass_batch(batch, task, self.device, cls_idx=0)

                # performing data replay on all previous tasks
                if data_replay_freq != -1 and i%data_replay_freq == 0:
                    self.memory_replay(tasks_to_generators, cls_idx=0)
                ######################

            # eval only on curr task (others are static)
            val_scores = self.eval_on_tasks(val_scores, split='val', prompt_tuning=prompt_tuning, original_task_id=task_id, tasks_to_eval=[task])
            if self.early_stopping:
                self.update_best_model(task, val_scores, tasks_to_eval=[task]) # update best model based on curr task acc improvement

        return val_scores



    def continual_training(self,
                           #tasks=[],
                           num_epochs=5,
                           data_replay_freq=-1,
                           prompt_tuning=True,
                           prompt_init='None',
                           save_prompt_path='None',
                           save_results_path='None',
                           ):
        results_dict = {}
        print('Continual training')
        if self.trainer.prefix_MLPs != None and prompt_tuning:
            cl_params = ['mlp', 'head']
        else:
            cl_params = ['head']

        for i, task in enumerate(self.task_list):
            self.trainer.freeze_unfreeze_mlps([x for x in self.task_list if x!=task], blocks=cl_params, requires_grad=False) # freezing MLPs & head for all tasks
            self.trainer.freeze_unfreeze_mlps([task], blocks=cl_params, requires_grad=True) # unfreezing current task MLP & head

            print('\n\nTASK ', task)
            val_scores = self.train_on_one_task(task,
                                                num_epochs=num_epochs,
                                                data_replay_freq=data_replay_freq,
                                                prompt_tuning=prompt_tuning)
            results_dict[i] = val_scores
            # loading the best model across all epochs (based on val acc)
            # in case of early stopping
            if self.early_stopping:
                self.trainer.model.load_state_dict(deepcopy(self.best_model))

            if prompt_tuning:
                # update wte matrix so we don't override the learned prompt with non-trained model
                self.trainer.update_baseline_model_emb()
                if save_prompt_path != 'None':
                    self.trainer.save_curr_task_emb(task_idx_curr=i, save_path=os.path.join(save_prompt_path, 'prompt_'+self.task_list[i]))
                if prompt_init != 'None' and task != self.task_list[-1]: # smart prompt initialization for the next prompts
                    print('Initializing new task prompt from the currently finished task')
                    self.trainer.init_new_prompt(task_idx_curr=i)

            if self.do_repeats:
                # overriding attention mask for the next task
                self.change_attention_mask_for_tasks(i+1)

            if save_results_path != 'None':
                test_scores = {x: [] for x in list(self.tasks_data_dict)}
                test_scores = self.eval_on_tasks(test_scores, split='test', prompt_tuning=prompt_tuning, original_task_id=len(self.task_list)-1)
                results_dict['test'] = test_scores
                np.save(os.path.join(save_results_path, 'results_dict_'+str(task)+'.npy'), results_dict)

        # final eval on test set
        test_scores = {x: [] for x in list(self.tasks_data_dict)}
        test_scores = self.eval_on_tasks(test_scores, split='test', prompt_tuning=prompt_tuning, original_task_id=len(self.task_list)-1)
        results_dict['test'] = test_scores
        return results_dict



    def multi_task_training(self, num_epochs=5, cls_idx=0):
        tasks_data_dict = self.tasks_data_dict
        val_scores = {x: [] for x in list(tasks_data_dict)}
        # getting index of the largest dataset (other datasets will be cycled)
        task_lengths = [len(tasks_data_dict[t]['train'])*self.batch_size for t in list(tasks_data_dict)]
        idx_biggest_task = np.argmax(task_lengths)
        n_tasks = len(list(tasks_data_dict))

        results_dict = {}
        val_scores = {x: [] for x in list(self.tasks_data_dict)}
        device = self.device

        for epoch in range(num_epochs):
            print(epoch)

            dataloaders_list = [tasks_data_dict[t]['train'] if j==idx_biggest_task else cycle(tasks_data_dict[t]['train']) \
                                for j, t in enumerate(tasks_data_dict)]
            mlt_dataloader = zip(*dataloaders_list)

            max_task = np.max([len(tasks_data_dict[t]['train']) for t in list(tasks_data_dict)])
            pbar = tqdm(total=max_task)
            for i, batch_combined in enumerate(mlt_dataloader):
                loss_combined = 0

                for task_num in range(n_tasks):
                    batch = {k: v.to(device) for k, v in batch_combined[task_num].items()}
                    loss = self.trainer.pass_batch(batch, list(tasks_data_dict)[task_num], self.device, cls_idx=cls_idx, only_output_loss=True)
                    loss_combined += loss

                loss_combined.backward()

                # only allowing updates for added special token if required
                if self.trainer.freeze_weights == 1 and self.trainer.freeze_except == 'word_embeddings':
                    k = len(trainer.special_tokens_list)
                    model.bert.embeddings.word_embeddings.weight.grad[:-k] = 0
                    #model.bert.embeddings.word_embeddings.weight.grad[:-1] = 0

                self.trainer.optimizer.step()
                self.trainer.scheduler.step()
                self.trainer.optimizer.zero_grad()
                pbar.update(1)

            val_scores = self.eval_on_tasks(val_scores, prompt_tuning=False, original_task_id=None)
            results_dict[epoch] = val_scores
            pbar.close()

        # final eval on test set
        test_scores = {x: [] for x in list(self.tasks_data_dict)}
        test_scores = self.eval_on_tasks(test_scores, split='test', prompt_tuning=False, original_task_id=None)
        results_dict['test'] = test_scores

        return results_dict
