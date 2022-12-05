import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse
import matplotlib.pyplot as plt

import dataset_utils, model_utils, continual_learning_utils
from itertools import cycle
from transformers import AdamW
from copy import deepcopy
from datasets import load_metric

from transformers import AdamW, get_constant_schedule_with_warmup

def change_string(str):
    #creating negative samples for NSP by randomly splitting positive samples
    #and swapping two halves
    if 102 in str:
        str.remove(102)
    if 102 in str:
        str.remove(102)

    len1 = len(str)
    if len1 == 1:
        cut = 1
    else:
        cut = np.random.randint(1, len1)
    str = str[cut:] + [102] + str[:cut] + [102]
    return str


def get_permutation_batch(src, src_mask, device, seq_len=512):
    #create negative samples for Next Sentence Prediction
    batch_size = src.size(0)
    length = src.size(1)
    dst = []
    dst_mask = []
    lbl = []
    for i in range(batch_size):
        cur = src[i]
        mask = src_mask[i].tolist()
        first_pad = (cur.tolist() + [0]).index(0)
        cur = cur[1:first_pad].tolist()
        cur = change_string(cur)
        lbl.append(1)

        padding = [0] * (length - len(cur) - 1)
        inp = torch.tensor([101] + cur + padding)
        dst.append(inp[:seq_len])
        dst_mask.append(torch.tensor(mask))
    return torch.stack(dst).to(device), torch.stack(dst_mask).to(device), torch.tensor(lbl).to(device)



def compute_class_offsets(tasks, task_classes):
    '''
    :param tasks: a list of the names of tasks, e.g. ["amazon", "yahoo"]
    :param task_classes:  the corresponding numbers of classes, e.g. [5, 10]
    :return: the class # offsets, e.g. [0, 5]
    Here we merge the labels of yelp and amazon, i.e. the class # offsets
    for ["amazon", "yahoo", "yelp"] will be [0, 5, 0]
    '''
    task_num = len(tasks)
    offsets = [0] * task_num
    prev = -1
    total_classes = 0
    for i in range(task_num):
        if tasks[i] in ["amazon", "yelp_review_full"]:
            if prev == -1:
                prev = i
                offsets[i] = total_classes
                total_classes += task_classes[i]
            else:
                offsets[i] = offsets[prev]
        else:
            offsets[i] = total_classes
            total_classes += task_classes[i]
    return total_classes, offsets



def pass_batch_reg(self, batch, device, cls_idx=0, only_output_loss=False):
    #if self.cls_idx_override!=None:
        #cls_idx = self.cls_idx_override

    model = self.model
    optimizer = self.optimizer
    scheduler = self.scheduler
    tokenizer = self.tokenizer

    batch = {k: v.to(device) for k, v in batch.items()}

    out = model.bert(**{'input_ids': batch['input_ids'],
                       'attention_mask': batch['attention_mask'],
                       'token_type_ids': batch['token_type_ids'],
                        })
    cls_output = out.last_hidden_state[:,cls_idx,:].to(device)

    if only_output_loss:
        # just returning the loss for subsequent operations (e.g. sum)
        # outputs = model.heads[task](outputs = out,
        #                         cls_output = cls_output,
        #                         return_dict = True,
        #                         labels = batch['labels'])

        # loss = outputs.loss
        #return loss, cls_output
        return cls_output

    else:
        # performing optimization step here
        loss.backward()
        # only allowing updates for added special token if required
        if self.freeze_weights == 1 and self.freeze_except == 'word_embeddings':
            k = len(self.special_tokens_list)
            model.bert.embeddings.word_embeddings.weight.grad[:-k] = 0

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()



class Predictor(torch.nn.Module):
    def __init__(self, num_class, hidden_size):
        super(Predictor, self).__init__()

        self.num_class = num_class

        self.dis = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, self.num_class)
        )

    def forward(self, z):
        return self.dis(z)



class ContinualLearnerIDBR:
    def __init__(self,
                 model_name,
                 task_list,
                 batch_size=8,
                 select_k_per_class=-1,
                 memory_perc=0,
                 #block_attn=0,
                 freeze_weights=0,
                 freeze_except='word_embeddings',
                 lr=3e-5, #2e-5
                 seq_len=512,
                 cls_idx_override=None,
                 early_stopping=True,
                 offsets=[],
                 total_classes=-1,
                 hidden_size=128,
                 tasks_data_dict=None,
                 regcoe=0.5,
                 regcoe_rply=5.0,
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
            'amazon': 5,
            'dbpedia_14': 14,
            'dbpedia': 14,

            'yelp': 5,
            'ag': 4,
            'yahoo': 10,
        }
        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue = ['cb', 'copa', 'wic', 'wsc', 'boolq', 'record', 'multirc']

        self.task_list = task_list
        num_labels_list = [self.task_to_num_labels[t] for t in self.task_list]
        self.total_classes, self.offsets = compute_class_offsets(self.task_list, num_labels_list)

        self.num_labels = [self.task_to_num_labels[t] for t in self.task_list]
        self.freeze_weights = freeze_weights
        self.lr = lr
        self.task_learning_rate = 5e-4
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cls_idx_override = cls_idx_override
        self.select_k_per_class = select_k_per_class
        self.memory_perc = memory_perc
        self.freeze_except = freeze_except
        self.early_stopping = early_stopping

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_name = model_name #"bert-base-uncased"
        self.trainer = model_utils.ModelForCL(self.model_name,
                                              tasks=self.task_list,
                                              num_labels=self.num_labels,
                                              #blockwise_causal_attention= (block_attn==1),
                                              freeze_weights= (self.freeze_weights==1),
                                              freeze_except=self.freeze_except,
                                              lr=self.lr,
                                              num_repeats=0, #self.num_repeats,
                                              max_length=self.seq_len, # max sequence length in #tokens
                                              cls_idx_override=self.cls_idx_override,
                                              )
        self.trainer.pass_batch_reg = lambda batch, device, cls_idx, only_output_loss: \
                                      pass_batch_reg(self.trainer, batch, device, cls_idx, only_output_loss)
        #self.trainer.pass_batch_reg = MethodType(lambda batch, task, device, cls_idx, only_output_loss:
        #                                         pass_batch_reg(batch, task, device,
        #                                                        cls_idx=cls_idx, only_output_loss=only_output_loss))

        #self.trainer.model.add_classification_head('giant', num_labels=total_classes)
        self.trainer.model.to(self.device) # model to cuda
        self.tokenizer = self.trainer.tokenizer
        if tasks_data_dict==None:
            self.tasks_data_dict = self.get_tasks_data_dict(self.select_k_per_class, memory_perc=self.memory_perc)
        else:
            print('Data is ready ', list(tasks_data_dict))
            self.tasks_data_dict = tasks_data_dict
        #### ADDING REG UTILS ####
        self.base_model = deepcopy(self.trainer.model) # bert before training

        self.hidden_size = hidden_size
        self.General_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        self.Specific_Encoder = nn.Sequential(
            nn.Linear(768, self.hidden_size),
            nn.Tanh()
        )

        n_class = self.total_classes
        self.cls_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, n_class)
        )

        n_tasks = len(self.task_list)
        self.task_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, n_tasks)
        )

        self.cls_CR = torch.nn.CrossEntropyLoss()
        self.predictor = Predictor(2, hidden_size=self.hidden_size).to(self.device) # NSP loss predictor
        self.nsp_CR = torch.nn.CrossEntropyLoss()

        if self.early_stopping:
            self.best_model = deepcopy(self.trainer.model.state_dict()) # saving best model
            self.best_GenEnc = deepcopy(self.General_Encoder.state_dict())
            self.best_SpeEnc = deepcopy(self.Specific_Encoder.state_dict())
            self.best_cls_classifier = deepcopy(self.cls_classifier.state_dict())
            self.best_task_classifier = deepcopy(self.task_classifier.state_dict())
            self.best_predictor = deepcopy(self.predictor.state_dict())
            self.best_acc = 0.0 # best avg accuracy on seen tasks

        self.trainer.optimizer = AdamW(
            [
                {"params": self.trainer.model.bert.parameters(), "lr": self.lr, "weight_decay": 0.01},
                {"params": self.General_Encoder.parameters(), "lr": self.lr, "weight_decay": 0.01},
                {"params": self.Specific_Encoder.parameters(), "lr": self.lr, "weight_decay": 0.01},
                {"params": self.cls_classifier.parameters(), "lr": self.lr, "weight_decay": 0.01},
                {"params": self.task_classifier.parameters(), "lr": self.task_learning_rate, "weight_decay": 0.01},
            ]
        )

        self.trainer.optimizer_P = AdamW(
            [
                {"params": self.predictor.parameters(), "lr": self.lr, "weight_decay": 0.01},
            ]
        )

        self.trainer.scheduler = get_constant_schedule_with_warmup(self.trainer.optimizer, 1000)
        self.trainer.scheduler_P = get_constant_schedule_with_warmup(self.trainer.optimizer_P, 1000)

        self.regcoe = regcoe
        self.regcoe_rply = regcoe_rply

        ##### ####### ##### ####


    # def get_tasks_data_dict(self, k, memory_perc=0):
    #     # if k==-1: use all data, otherwise use k examples from class
    #     trainer = self.trainer
    #     tasks_data_dict = {}

    #     k_val = -1 if k==-1 else int(k*0.15)
    #     for j, task in enumerate(self.task_list):
    #         tasks_data_dict[task] = {}
    #         print(task)
    #         du = dataset_utils.Dataset(task=task, tokenizer=trainer.tokenizer, idbr_preprocessing=True) # turn on idbr_preprocessing flag
    #         data_params = {'repeats': trainer.num_repeats,
    #                        'batch_size': self.batch_size,
    #                        'max_length': trainer.max_length,
    #                        'label_offset': self.offsets[j],
    #                        #'select_k_per_class': k
    #                        }
    #         benchmark = 'glue' if task in self.glue_datasets else None
    #         val_split = 'validation' if task in self.glue_datasets else 'test'

    #         dataloader_train = du.get_dataset(benchmark=benchmark, split='train', select_k_per_class=k, **data_params)
    #         if memory_perc>0:
    #             k_mem = int(len(dataloader_train)*memory_perc)
    #             dataloader_mem = du.get_dataset(benchmark=benchmark, split='train',
    #                                             select_k_per_class=k_mem, **data_params)

    #         if k!=-1:
    #             if task == 'dbpedia': k_val = int(k*0.1)
    #             elif task != 'dbpedia': k_val = int(k*0.15)
    #         dataloader_val, dataloader_test = du.get_dataset(benchmark=benchmark, split=val_split,
    #                                                          select_k_per_class=k_val, return_test_subset=True,
    #                                                          **data_params)

    #         tasks_data_dict[task]['train'] = dataloader_train
    #         if memory_perc>0: tasks_data_dict[task]['train_mem'] = dataloader_mem # for data replay
    #         tasks_data_dict[task]['val'] = dataloader_val
    #         tasks_data_dict[task]['test'] = dataloader_test

    #     return tasks_data_dict

    def get_tasks_data_dict(self, k, memory_perc=0):
        # if k==-1: use all data, otherwise use k examples from class
        trainer = self.trainer
        tasks_data_dict = {}
        current_task_progressive_prompt = []

        k_val = -1 if k==-1 else max(int(k*0.15), 500)
        for task in self.task_list:
            tasks_data_dict[task] = {}
            print(task)
            current_task_progressive_prompt += trainer.prefix_tokens_list[task]
            print(current_task_progressive_prompt)
            du = dataset_utils.Dataset(task=task, tokenizer=trainer.tokenizer)
            tid = self.task_list.index(task) # task id
            data_params = {'repeats': 0, #trainer.num_repeats,
                           'batch_size': self.batch_size,
                           'max_length': trainer.max_length,
                           'prefix_tokens_list': [],
                           'prefix_len': 0, # prompt * task_quantity
                           'do_repeats': False, # only applies to the first task in repeated set-up (if we format it according to repeats)
                           }
            benchmark = 'glue' if task in self.glue_datasets else 'super_glue' if task in self.superglue else None
            val_split = 'validation' if (task in self.glue_datasets or task in self.superglue) else 'test'
            dataloader_train = du.get_dataset(benchmark=benchmark, split='train', select_k_per_class=k, **data_params)
            if memory_perc>0:
                k_mem = max( int(len(dataloader_train)*self.batch_size*memory_perc), 1) # no less than 1
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


    # returns metric corresponding to the task
    def task_to_metric_key(self, task):
        if task not in self.glue_datasets:
            return 'accuracy'

        if task in ['qqp', 'mrpc']:
            return 'f1'

        elif 'mnli' in task or task == 'cola':
            return 'matthews_correlation'

        else:
            return 'accuracy'


    def eval_repr_split(self, trainer, dataloader_val, task, metric, cls_idx=0):
        if trainer.cls_idx_override!=None:
            cls_idx = trainer.cls_idx_override
        model = trainer.model
        tokenizer = trainer.tokenizer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval().to(device)

        if metric==None:
            #if task in self.glue_datasets:
            if task in ['qqp', 'mrpc', 'cola'] or 'mnli' in task:
                metric = load_metric('glue', task)
            else:
                metric = load_metric('accuracy')
        print(metric.name)
        #if trainer.num_repeats>=1:
        #    pos = trainer.get_position_ids(dataloader_val, device)

        for i, batch in enumerate(tqdm(dataloader_val)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                inp_dict = {'input_ids': batch['input_ids'],
                            'attention_mask': batch['attention_mask'],
                            'token_type_ids': batch['token_type_ids'],
                            }

                out = model.bert(**inp_dict)
                cls_output = out.last_hidden_state[:,cls_idx,:].to(device)
#               outputs = model.heads[task](outputs = out,
#                                             cls_output = cls_output,
#                                             return_dict = True,
#                                             labels = batch['labels'])

#               predictions = torch.argmax(outputs.logits, dim=-1)
                general_features = self.General_Encoder(cls_output)
                specific_features = self.Specific_Encoder(cls_output)
                features = torch.cat([general_features, specific_features], dim=1)
                cls_pred = self.cls_classifier(features)

                # Calculate classification loss
                _, pred_cls = cls_pred.max(1)
                #y = batch['labels']
                #correct_cls = pred_cls.eq(y.view_as(pred_cls)).sum().item()
            metric.add_batch(predictions=pred_cls, references=batch['labels'])

        try:
            result = metric.compute()
            metric_key = self.task_to_metric_key(task) # we want to return value float (not dict metric -> value)
            result = result[metric_key]
        except: result=0.0 # could not compute (maybe all labels are the same and f1 not computing)
        return result


    def eval_on_tasks(self, val_scores, cls_idx=0, split='val', repr_split=True):
        self.trainer.model.eval()
        self.cls_classifier.eval()
        self.task_classifier.eval()
        self.General_Encoder.eval()
        self.Specific_Encoder.eval()
        self.predictor.eval()

        for task in list(self.tasks_data_dict):
            dataloader_val = self.tasks_data_dict[task][split]
            if repr_split:
                result = self.eval_repr_split(self.trainer, dataloader_val, task, None, cls_idx)
            else:
                result = self.trainer.eval(dataloader_val, 'giant', None, cls_idx)
            print(task, ' result = ',result)
            val_scores[task].append(result)

        return val_scores



    def update_best_model(self, val_scores, new_task = False):
        #idx = list(self.tasks_data_dict).index(curr_task) # look tasks up to curr task
        #seen_tasks = [t for t in list(self.tasks_data_dict)[:idx+1]]
        seen_tasks = list(self.tasks_data_dict)
        avg_acc = np.mean([val_scores[task][-1] for task in seen_tasks])
        # only update if we are starting a new task OR if acc gets better
        if avg_acc > self.best_acc: # or new_task:
            print('NEW BEST MODEL acc=', avg_acc)
            self.best_acc = avg_acc
            self.best_model = deepcopy(self.trainer.model.state_dict())
            self.best_GenEnc = deepcopy(self.General_Encoder.state_dict())
            self.best_SpeEnc = deepcopy(self.Specific_Encoder.state_dict())
            self.best_cls_classifier = deepcopy(self.cls_classifier.state_dict())
            self.best_task_classifier = deepcopy(self.task_classifier.state_dict())
            self.best_predictor = deepcopy(self.predictor.state_dict())


    def get_loss_from_representation(self, bert_cls_embedding, batch,
                                     regspe, reggen, tskcoe, nspcoe, disen,
                                     task, data_replay_freq,
                                     cls_idx=0):
        if self.cls_idx_override!=None:
            cls_idx = self.cls_idx_override

        task_id = list(self.tasks_data_dict).index(task)
        replay = data_replay_freq!=-1

        if disen:
            x, mask = batch['input_ids'], batch['attention_mask']
            p_x, p_mask, p_lbl = get_permutation_batch(x, mask, self.device, seq_len=self.seq_len)

            #x = torch.cat([x, p_x], dim=0)
            #mask = torch.cat([mask, p_mask], dim=0)
            r_lbl = torch.zeros_like(p_lbl)
            nsp_lbl = torch.cat([r_lbl, p_lbl], dim=0)

            y = torch.cat([batch['labels'], batch['labels']], dim=0)
            t = torch.tensor([task_id]*self.batch_size*2).to(self.device) # correct task ids

            p_out = self.trainer.model.bert(**{'input_ids': p_x,
                                               'attention_mask': p_mask,
                                               'token_type_ids': batch['token_type_ids'],
                                                })
            p_cls_output = p_out.last_hidden_state[:,cls_idx,:].to(self.device)
            bert_cls_embedding = torch.cat([bert_cls_embedding, p_cls_output], dim=0)
        else:
            y = batch['labels']
            t = torch.tensor([task_id]*self.batch_size).to(self.device) # correct task ids

        general_features = self.General_Encoder(bert_cls_embedding)
        specific_features = self.Specific_Encoder(bert_cls_embedding)

        features = torch.cat([general_features, specific_features], dim=1)
        cls_pred = self.cls_classifier(features)
        task_pred = self.task_classifier(specific_features)

        # Calculate classification loss
        _, pred_cls = cls_pred.max(1)
        #y = batch['labels']
        #correct_cls = pred_cls.eq(y.view_as(pred_cls)).sum().item()
        cls_loss = self.cls_CR(cls_pred, y)


        reg_loss = torch.tensor(0.0).to(self.device)
        task_loss = torch.tensor(0.0).to(self.device)
        nsp_loss = torch.tensor(0.0).to(self.device)

        if task_id >0 and regspe>0 and reggen>0:
            # Calculate reg loss
            base_out = self.base_model.bert(**{'input_ids': batch['input_ids'],
                                               'attention_mask': batch['attention_mask'],
                                               'token_type_ids': batch['token_type_ids'],
                                                })
            base_emb = base_out.last_hidden_state[:,cls_idx,:].to(self.device)

            old_g_fea = self.General_Encoder(base_emb)
            old_s_fea = self.Specific_Encoder(base_emb)
            lim = old_s_fea.shape[0] # previously was self.batch_size
            reg_loss += regspe * torch.nn.functional.mse_loss(specific_features[:lim], old_s_fea) + \
                        reggen * torch.nn.functional.mse_loss(general_features[:lim],  old_g_fea)
            if replay and task_id > 0:
                reg_loss *= self.regcoe_rply
            elif not replay and task_id > 0:
                reg_loss *= self.regcoe

        # Calculate task loss only when in replay batch
        if task_id > 0 and replay and tskcoe>0:
            task_pred = task_pred[:, :task_id + 1]
            _, pred_task = task_pred.max(1)
            # correct_task = pred_task.eq(t.view_as(pred_task)).sum().item()
            task_loss += tskcoe * self.cls_CR(task_pred, t[:task_pred.shape[0]])

        # Calculate nsp loss
        if disen and nspcoe>0:
            nsp_output = self.predictor(general_features)
            nsp_loss += nspcoe * self.nsp_CR(nsp_output, nsp_lbl)
            #_, nsp_pred = nsp_output.max(1)
            #nsp_correct = nsp_pred.eq(nsp_lbl.view_as(nsp_pred)).sum().item()
            #nsp_acc = nsp_correct * 1.0 / (batch_size * 2.0)

        loss = cls_loss + reg_loss + task_loss + nsp_loss
        return loss


    def train_on_one_task(self,
                          task,
                          data_replay_freq = -1, # if -1 no data replay, else replay after N samples
                          num_epochs = 5,
                          regspe=0.5,
                          reggen=0.5,
                          tskcoe=1.0,
                          nspcoe=1.0,
                          disen=True):

        val_scores = {x: [] for x in list(self.tasks_data_dict)}
        device = self.device

        self.Specific_Encoder.to(self.device)
        self.General_Encoder.to(self.device)
        self.cls_classifier.to(self.device)
        self.task_classifier.to(self.device)
        self.trainer.model.to(self.device)
        self.predictor.to(self.device)

        optimizer = self.trainer.optimizer
        scheduler = self.trainer.scheduler

        for epoch in range(num_epochs):
            print(epoch)
            self.trainer.model.train()
            self.cls_classifier.train()
            self.task_classifier.train()
            self.General_Encoder.train()
            self.Specific_Encoder.train()
            self.predictor.train()

            if data_replay_freq != -1:
                print('Creating generators for previous tasks ...')
                tasks_to_generators = {}
                curr_task_num = list(self.tasks_data_dict).index(task)
                for idx in np.arange(curr_task_num):
                    prev_task = list(self.tasks_data_dict)[idx]
                    print(prev_task)
                    tasks_to_generators[prev_task] = iter(self.tasks_data_dict[prev_task]['train_mem'])

            for i, batch in enumerate(tqdm(self.tasks_data_dict[task]['train'])):
                batch = {k: v.to(device) for k, v in batch.items()}
                # we will ignore the default loss
                bert_cls_embedding = self.trainer.pass_batch_reg(batch, self.device,
                                                                    cls_idx=0, only_output_loss=True)
                bert_cls_embedding = bert_cls_embedding.to(self.device)

                #### ADDING REG UTILS ####
                loss = self.get_loss_from_representation(bert_cls_embedding, batch,
                                                         regspe, reggen, tskcoe, nspcoe, disen,
                                                         task, data_replay_freq)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ####   ####   ####

                # performing data replay on all previous tasks
                if data_replay_freq != -1 and i%data_replay_freq == 0:
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

                        b = {k: v.to(device) for k, v in b.items()}
                        #self.trainer.pass_batch(b, 'giant', self.device, cls_idx=0)
                        bert_cls_embedding = self.trainer.pass_batch_reg(b, self.device,
                                                                            cls_idx=0, only_output_loss=True)
                        bert_cls_embedding = bert_cls_embedding.to(self.device)
                        loss = self.get_loss_from_representation(bert_cls_embedding, b,
                                                                 regspe, reggen, tskcoe, nspcoe, disen,
                                                                 task, data_replay_freq)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                ######################

                #if i%250 == 0 and i>0: # check val accuracy every 250 iterations
            val_scores = self.eval_on_tasks(val_scores, cls_idx=0)
            if self.early_stopping:
                self.update_best_model(val_scores)

        return val_scores


    def continual_training(self,
                           #tasks=[],
                           num_epochs=5,
                           data_replay_freq=-1,
                           regspe=0.5,
                           reggen=0.5,
                           tskcoe=1.0,
                           nspcoe=1.0,
                           disen=True):
        results_dict = {}
        print('Continual training')
        for i, task in enumerate(self.task_list):
            if i>0 and self.early_stopping:
                self.update_best_model(val_scores, new_task=True)

            print('\n\nTASK ', task)
            val_scores = self.train_on_one_task(task,
                                                num_epochs=num_epochs,
                                                data_replay_freq=data_replay_freq,
                                                regspe=regspe,
                                                reggen=reggen,
                                                tskcoe=tskcoe,
                                                nspcoe=nspcoe,
                                                disen=disen)
            results_dict[i] = val_scores
            # loading the best model across all epochs (based on val acc)
            # in case of early stopping
            if self.early_stopping:
                self.trainer.model.load_state_dict(deepcopy(self.best_model))
                self.General_Encoder.load_state_dict(deepcopy(self.best_GenEnc))
                self.Specific_Encoder.load_state_dict(deepcopy(self.best_SpeEnc))
                self.cls_classifier.load_state_dict(deepcopy(self.best_cls_classifier))
                self.task_classifier.load_state_dict(deepcopy(self.best_task_classifier))
                self.predictor.load_state_dict(deepcopy(self.best_predictor))

            # update regularization model
            self.base_model = deepcopy(self.trainer.model) # bert before training

        # final eval on test set
        test_scores = {x: [] for x in list(self.tasks_data_dict)}
        test_scores = self.eval_on_tasks(test_scores, cls_idx=0, split='test')
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
        device = self.device

        self.cls_classifier.to(self.device)
        self.trainer.model.to(self.device)
        self.task_classifier.to(self.device)
        self.General_Encoder.to(self.device)
        self.Specific_Encoder.to(self.device)
        self.predictor.to(self.device)

        for epoch in range(num_epochs):
            print(epoch)
            self.trainer.model.train()
            self.cls_classifier.train()
            self.task_classifier.train()
            self.General_Encoder.train()
            self.Specific_Encoder.train()
            self.predictor.train()

            dataloaders_list = [tasks_data_dict[t]['train'] if j==idx_biggest_task else cycle(tasks_data_dict[t]['train']) \
                                for j, t in enumerate(tasks_data_dict)]
            mlt_dataloader = zip(*dataloaders_list)

            max_task = np.max([len(tasks_data_dict[t]['train']) for t in list(tasks_data_dict)])
            pbar = tqdm(total=max_task)
            for i, batch_combined in enumerate(mlt_dataloader):
                loss_combined = 0

                for task_num in range(n_tasks):
                    task = list(self.tasks_data_dict)[task_num]
                    batch = {k: v.to(device) for k, v in batch_combined[task_num].items()}
                    bert_cls_embedding = self.trainer.pass_batch_reg(batch, device, cls_idx=cls_idx, only_output_loss=True)
                    bert_cls_embedding = bert_cls_embedding.to(device)
                    loss = self.get_loss_from_representation(bert_cls_embedding, batch,
                                                             0, 0, 0, 0, False, task, -1)
                    loss_combined += loss

                loss_combined.backward()

                # only allowing updates for added special token if required
                #if self.trainer.freeze_weights == 1 and self.trainer.freeze_except == 'word_embeddings':
                    #k = len(trainer.special_tokens_list)
                    #model.bert.embeddings.word_embeddings.weight.grad[:-k] = 0
                    #model.bert.embeddings.word_embeddings.weight.grad[:-1] = 0

                self.trainer.optimizer.step()
                self.trainer.scheduler.step()
                self.trainer.optimizer.zero_grad()
                pbar.update(1)

            val_scores = self.eval_on_tasks(val_scores, cls_idx=cls_idx)
            if self.early_stopping:
                self.update_best_model(val_scores)

            results_dict[epoch] = val_scores
            pbar.close()

        # final eval on test set
        if self.early_stopping:
            self.trainer.model.load_state_dict(deepcopy(self.best_model))
            self.General_Encoder.load_state_dict(deepcopy(self.best_GenEnc))
            self.Specific_Encoder.load_state_dict(deepcopy(self.best_SpeEnc))
            self.cls_classifier.load_state_dict(deepcopy(self.best_cls_classifier))
            self.task_classifier.load_state_dict(deepcopy(self.best_task_classifier))
            self.predictor.load_state_dict(deepcopy(self.best_predictor))
        test_scores = {x: [] for x in list(self.tasks_data_dict)}
        test_scores = self.eval_on_tasks(test_scores, cls_idx=cls_idx, split='test')
        results_dict['test'] = test_scores

        return results_dict
