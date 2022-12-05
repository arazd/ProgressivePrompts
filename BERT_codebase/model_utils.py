from transformers import AutoTokenizer, AutoAdapterModel
import numpy as np
from transformers.adapters import PrefixTuningConfig
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING, BertModel, BertPreTrainedModel
#from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
#from transformers.context import AdapterSetup
from transformers.adapters.heads import (
    BertStyleMaskedLMHead,
    BiaffineParsingHead,
    CausalLMHead,
    ClassificationHead,
    ModelWithFlexibleHeadsAdaptersMixin,
    MultiLabelClassificationHead,
    MultipleChoiceHead,
    QuestionAnsweringHead,
    TaggingHead,
)
from transformers.adapters import BertAdapterModel
from transformers import AdamW, get_constant_schedule_with_warmup
from types import MethodType # to update attention calculation of BERT
from torch import Tensor, device, nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
from datasets import load_metric

from copy import deepcopy

glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']

def get_mask_arr(k = 2, block_size = 100, device = None):
    n = k+1 # number of "stacked" input blocks
    mask_arr = np.ones([block_size*n, block_size*n])

    for i in range(n):
        for m in range(i+1, n):
            mask_arr[i*block_size : (i+1)*block_size,
                     m*block_size : (m+1)*block_size] = 0
    return torch.Tensor(mask_arr).to(device)



# overriding attention mask for BERT
# usage: self.get_extended_attention_mask(attention_mask, input_shape, device)
def get_extended_attention_mask2(
   self, attention_mask: Tensor, input_shape: Tuple[int], device: device = None,
   k = 2, block_size = 100, blockwise_causal_mask = None,
) -> Tensor:

    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder:
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]

        # OUR MODIFICATION FOR CUSTOM MASK
        if k != None and block_size != None:
            #print('applying blockwise attention')
            # for blockwise_causal_mask,
            # we broadcast [seq_len, seq_len] to [batch_size, num_heads, seq_length, seq_length]
            blockwise_causal_mask = get_mask_arr(k = k, block_size = block_size, device = device)[None, None, :, :]

        if blockwise_causal_mask != None:
            #print('blockwise_causal_mask ', blockwise_causal_mask.shape)
            #print('extended_attention_mask ', extended_attention_mask.shape)
            with torch.no_grad():
                extended_attention_mask = blockwise_causal_mask * extended_attention_mask.to(device)
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask



class ModelForCL:
    def __init__(self,
                 model_name,
                 tasks=['cola'],
                 num_labels=[2],
                 #blockwise_causal_attention=False,
                 prefix_len=0,
                 freeze_weights=False,
                 freeze_except='word_embeddings',
                 lr=2e-5,
                 num_repeats=0,
                 max_length=150, # max sequence length in #tokens
                 cls_idx_override=None,
                 prefix_MLPs=None,
                 same_prompt=False, # whether to use the same prompt for all tasks
                 ):
        self.model_name = model_name # "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.prefix_MLPs = prefix_MLPs

        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.same_prompt = same_prompt
        #self.current_task = tasks[0]
        self.num_repeats = num_repeats
        self.max_length = max_length
        self.cls_idx_override = cls_idx_override
        # overriding attention mechanism to make block-wise causal attention
        #self.blockwise_causal_attention = blockwise_causal_attention

        cls_tok = self.tokenizer.tokenize("[CLS]")
        self.cls_id = self.tokenizer.convert_tokens_to_ids(cls_tok)[0]

        # freezing weights except word embeddings
        self.freeze_except = freeze_except
        self.freeze_weights = freeze_weights
        if self.freeze_weights:
            self.do_freeze_weights(except_condition=freeze_except)

        #self.model.add_classification_head(self.current_task, num_labels=num_labels[0])
        for i in range(self.num_tasks):
            self.model.add_classification_head(self.tasks[i], num_labels=num_labels[i])

        # adding special prefix tokens (CHANGE for new tasks)
        self.prefix_len = prefix_len

        if prefix_len > 0:
            self.prefix_tokens_list = {}

            if self.same_prompt: # assume we have just 1 task (i.e. 1 prompt for each task)
                self.prefix_tokens_list[0] = self.add_prompt_tokens(prefix_len, prompt_name='PRE0_')
            else:
                for i in range(self.num_tasks):
                    # new prefix for each task
                    # Task 1 = PRE1_1, ... PRE1_30 ; Task 2 = PRE2_1, ... PRE2_30
                    self.prefix_tokens_list[self.tasks[i]] = self.add_prompt_tokens(prefix_len, prompt_name='PRE'+str(i)+'_')
        else:
            self.prefix_tokens_list = {self.tasks[i]: [] for i in range(self.num_tasks)} # empty prompt for each task

        #self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        params_group = [{"params": self.model.parameters(), "lr": lr, "weight_decay": 0.01},]
        if self.prefix_MLPs != None:
            for t in prefix_MLPs: # append parameters of each task MLP
                params_group.append({"params": self.prefix_MLPs[t].parameters(), "lr": lr, "weight_decay": 0.01})
        self.optimizer = AdamW(params_group)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, 1000)

        # save prompt emb for all tasks
        #self.saved_embs = deepcopy(self.model.bert.embeddings.word_embeddings.weight[:-self.prefix_len*self.num_tasks].cpu().detach().numpy())
        self.saved_embs = deepcopy(self.model.bert.embeddings.word_embeddings.weight.cpu().detach().numpy())



    def add_prompt_tokens(self, prefix_len, prompt_name='PRE'):
        tokenizer = self.tokenizer
        model = self.model
        N = model.bert.embeddings.word_embeddings.weight.shape[0] # wte shape before resize

        # tokens_list - ['[PRE1]', '[PRE2]', '[PRE3]']
        tokens_list = ['['+ prompt_name + str(i) + ']' for i in np.arange(1, prefix_len+1)]
        special_tokens_dict = {'additional_special_tokens': tokens_list}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.bert.resize_token_embeddings(len(tokenizer))

        model.bert.embeddings.word_embeddings.weight.requires_grad = False
        for i in range(len(tokens_list)):
            with torch.no_grad():
                # initialize pre1 as CLS token
                if i==0:
                    j = self.cls_id
                # initalize pre2, pre3 ... with random word embedding
                else:
                    j = np.random.randint(N)
                #model.bert.embeddings.word_embeddings.weight[N+i, :] = \
                #model.bert.embeddings.word_embeddings.weight[j]
                w = deepcopy(model.bert.embeddings.word_embeddings.weight[j].detach().cpu().numpy())
                model.bert.embeddings.word_embeddings.weight[N+i] = torch.from_numpy(w)
        model.bert.embeddings.word_embeddings.weight.requires_grad = True
        return tokens_list


    # freeze all weights except for word emb (or other condition specified)
    def do_freeze_weights(self, except_condition='word_embeddings'):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad == True and except_condition not in name:
                param.requires_grad = False

    # freeze / unfreeze MLPs for given tasks (when requires_grad==False then freezing)
    def freeze_unfreeze_mlps(self, tasks, blocks=['mlp', 'head'], requires_grad=False):
        for x in blocks: # we only freeze/unfreeze cls heads and MLPs for CL setting
            assert x in ['mlp', 'head']

        param_groups = []
        for name in blocks:
            if name=='mlp':
                assert self.prefix_MLPs != None
                param_groups.append(self.prefix_MLPs)
            if name=='head':
                param_groups.append(self.model.heads)

        for t in tasks:
            for p_group in param_groups:
                #for name, param in self.prefix_MLPs[t].named_parameters():
                for name, param in p_group[t].named_parameters():
                    if param.requires_grad != requires_grad:
                        param.requires_grad = requires_grad
                        param.grad = None # remove old gradient


    # overriding attention mask for blockwise causal attention:
    def override_attention_mask(self, num_repeats, repeat_length, device):
        blockwise_causal_mask = get_mask_arr(k=num_repeats, block_size=repeat_length, device=device)
        self.model.bert.get_extended_attention_mask = MethodType(lambda self, attention_mask, input_shape, device: \
                                                                 get_extended_attention_mask2(self, attention_mask, input_shape, device,
                                                                                              k = None,
                                                                                              block_size = None,
                                                                                              blockwise_causal_mask = blockwise_causal_mask,),
                                                                                              self.model.bert)
        return blockwise_causal_mask

    # get custom position ids for sentence with repeats
    # [cls] input     [cls1] input
    #   0   1 2 ... N   0    N+1 ...
    def get_position_ids(self, dataloader, device):
        tokenizer = self.tokenizer
        b = next(iter(dataloader))
        pos = list(range( len(b['input_ids'][0]) ))
        #pos[100] = 0
        for tok in self.special_tokens_list:
            tokenized = tokenizer.tokenize(tok)
            tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
            clsN_pos = int(torch.where(b['input_ids'][0] == tokenized_ids[0])[0].cpu())
            pos[clsN_pos] = 0

        pos = torch.tensor(pos)
        pos = torch.cat([pos.view(1,-1)]* len(b['input_ids']), axis=0).to(device)
        return pos


    def train(self, dataloader, task, epochs=5, cls_idx=100, dataloader_val=None, metric=None):
        #cls_idx=self.max_length # CHANGE FOR TASK 3+
        if self.cls_idx_override!=None:
            cls_idx = self.cls_idx_override
        print('Using CLS idx ', cls_idx)
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        tokenizer = self.tokenizer
        # fine-tuning with 2 repeats
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.train().to(device)

        if self.num_repeats>=1:
            pos = self.get_position_ids(dataloader, device)

        val_scores = []

        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}

                if self.num_repeats < 1 or cls_idx==0: ## num_repeats <1 or ==1?
                    outputs = model(**batch)

                else:
                    out = model.bert(**{'input_ids': batch['input_ids'],
                                      'attention_mask': batch['attention_mask'],
                                      'token_type_ids': batch['token_type_ids'],
                                      'position_ids': pos[:len(batch['input_ids'])],
                                       })
                    cls_output = out.last_hidden_state[:,cls_idx,:].to(device)
                    outputs = model.heads[task](outputs = out,
                                                cls_output = cls_output,
                                                return_dict = True,
                                                labels = batch['labels'])

                loss = outputs.loss
                loss.backward()

                # only allowing updates for added special token if required
                if self.freeze_weights == 1 and self.freeze_except == 'word_embeddings':
                    k = len(self.special_tokens_list)
                    model.bert.embeddings.word_embeddings.weight.grad[:-k] = 0
                    #model.bert.embeddings.word_embeddings.weight.grad[:-1] = 0
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if dataloader_val != None:
                result = self.eval(dataloader_val, self.current_task, metric, cls_idx)
                print('result = ',result)
                #metric_key = list(result)[0] # append results as value floats, instead of dict metric -> value
                metric_key = self.task_to_metric_key(task)
                val_scores.append(result[metric_key])

        return val_scores



    def pass_batch(self, batch, task, device, cls_idx=0, only_output_loss=False):
        if self.cls_idx_override!=None:
            cls_idx = self.cls_idx_override
        #print('Using CLS idx ', cls_idx)
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        tokenizer = self.tokenizer

        batch = {k: v.to(device) for k, v in batch.items()}

        out = model.bert(**{'input_ids': batch['input_ids'],
                           'attention_mask': batch['attention_mask'],
                           'token_type_ids': batch['token_type_ids'],
                           #'position_ids': pos[:len(batch['input_ids'])],
                            })
        cls_output = out.last_hidden_state[:,cls_idx,:].to(device)

        outputs = model.heads[task](outputs = out,
                                    cls_output = cls_output,
                                    return_dict = True,
                                    labels = batch['labels'])

        loss = outputs.loss

        if only_output_loss:
            # just returning the loss for subsequent operations (e.g. sum)
            return loss

        else:
            # performing optimization step here
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()



    def pass_batch_with_prompt(self, batch, task, device, prefix_len=None, cls_idx=0, custom_pos_ids=True):
        if self.cls_idx_override!=None:
            cls_idx = self.cls_idx_override # position of cls in sentence (usually 0)

        if prefix_len == None:
            prefix_len = self.prefix_len
        #print('Using CLS idx ', cls_idx)
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        tokenizer = self.tokenizer
        prefix_MLP = None if self.prefix_MLPs==None else self.prefix_MLPs[task]
        if self.same_prompt:
            task_idx = 0 # constast task idx bcs of shared prompt
        else:
            task_idx = self.tasks.index(task)

        batch_keys = ['input_ids', 'token_type_ids']
        if custom_pos_ids:
            batch_keys.append('position_ids') # loop through custom position ids
        emb = model.bert.embeddings(**{k: batch[k] for k in batch_keys})
        if prefix_len>0 and prefix_MLP != None:
            # WORKS FOR NO REPEATS CASE
            #emb[:, :prefix_len, :] = prefix_MLP(emb[:, :prefix_len, :].clone().to(device))
            pos1, pos2 = self.max_length + prefix_len*task_idx, self.max_length + prefix_len*(task_idx+1)
            emb[:, pos1:pos2, :] = prefix_MLP(emb[:, pos1:pos2, :].clone().to(device))

        extended_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(batch['attention_mask'],
                                                                                       batch['input_ids'].shape,
                                                                                       device=device)
        out = model.bert.encoder(emb, attention_mask=extended_attention_mask)
        cls_output = out.last_hidden_state[:,cls_idx,:].to(device)
        outputs = model.heads[task](outputs = out,
                                    cls_output = cls_output,
                                    return_dict = True,
                                    labels = batch['labels'])

        loss = outputs.loss
        loss.backward()

        # only allowing updates for added special token if required
        #if freeze_except == 'word_embeddings':
        if prefix_len>0:
            emb_size = model.bert.embeddings.word_embeddings.weight.shape[0]
            k1, k2 = emb_size - prefix_len * (self.num_tasks - task_idx), emb_size - prefix_len * (self.num_tasks - task_idx -1)
            model.bert.embeddings.word_embeddings.weight.grad[:k1] = 0
            model.bert.embeddings.word_embeddings.weight.grad[k2:] = 0

        optimizer.step()
        scheduler.step()


        #model.bert.embeddings.word_embeddings.weight[:-prefix_len] = torch.from_numpy(self.saved_embs)
        if prefix_len>0:
            model.bert.embeddings.word_embeddings.weight.requires_grad = False
            model.bert.embeddings.word_embeddings.weight[:k1] = torch.from_numpy(self.saved_embs[:k1]) # restore all emb except curr task
            model.bert.embeddings.word_embeddings.weight[k2:] = torch.from_numpy(self.saved_embs[k2:])
            model.bert.embeddings.word_embeddings.weight.requires_grad = True
        optimizer.zero_grad()



    def update_baseline_model_emb(self):
        # update word emb matrix for continual prompt tuning
        # so that we don't "forget" learned prompts during re-setting
        self.saved_embs = deepcopy(self.model.bert.embeddings.word_embeddings.weight.cpu().detach().numpy())


    # initialize new task prompt from previous task prompts
    def init_new_prompt(self, task_idx_curr):
        prefix_len = self.prefix_len
        model = self.model
        emb_size = model.bert.embeddings.word_embeddings.weight.shape[0]
        k1_curr, k2_curr = emb_size - prefix_len * (self.num_tasks - task_idx_curr), emb_size - prefix_len * (self.num_tasks - task_idx_curr -1)
        task_idx_next = task_idx_curr+1
        k1_next, k2_next = emb_size - prefix_len * (self.num_tasks - task_idx_next), emb_size - prefix_len * (self.num_tasks - task_idx_next -1)

        model.bert.embeddings.word_embeddings.weight.requires_grad = False
        model.bert.embeddings.word_embeddings.weight[k1_next:k2_next] = torch.from_numpy(self.saved_embs[k1_curr:k2_curr]) # init new task emb from curr task emb
        model.bert.embeddings.word_embeddings.weight.requires_grad = True


    def save_curr_task_emb(self, task_idx_curr, save_path):
        prefix_len = self.prefix_len
        model = self.model
        emb_size = model.bert.embeddings.word_embeddings.weight.shape[0]
        k1_curr, k2_curr = emb_size - prefix_len * (self.num_tasks - task_idx_curr), emb_size - prefix_len * (self.num_tasks - task_idx_curr -1)
        np.save(save_path, self.saved_embs[k1_curr:k2_curr]) # save np array with curr task emb


    # modifies output by passing soft prompt embs throught MLP
    def get_bert_output_with_prompt(self, batch, prefix_len, device):
        model = self.model
        emb = model.bert.embeddings(**{k: batch[k] for k in ['input_ids', 'token_type_ids']})
        emb[:, :prefix_len, :] = self.prefix_MLP(emb[:, :prefix_len, :].to(device))

        extended_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(batch['attention_mask'],
                                                                                        batch['input_ids'].shape,
                                                                                        device=device)
        out = model.bert.encoder(emb, attention_mask=extended_attention_mask)
        return out



    # returns metric corresponding to the task
    def task_to_metric_key(self, task):
        if task not in glue_datasets:
            return 'accuracy'

        if task in ['qqp', 'mrpc']:
            return 'f1'

        #elif 'mnli' in task or task == 'cola':
        elif task == 'cola':
            return 'matthews_correlation'

        else:
            return 'accuracy'



    def eval(self, dataloader_val, task, metric, cls_idx=100):
        #ls_idx=self.max_length # CHANGE FOR TASK 3+
        if self.cls_idx_override!=None:
            cls_idx = self.cls_idx_override
        model = self.model
        tokenizer = self.tokenizer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval().to(device)

        if metric==None:
            if task in glue_datasets:
                metric = load_metric('glue', task)
            else:
                metric = load_metric('accuracy')

        if self.num_repeats>=1:
            pos = self.get_position_ids(dataloader_val, device)

        for i, batch in enumerate(tqdm(dataloader_val)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                inp_dict = {'input_ids': batch['input_ids'],
                            'attention_mask': batch['attention_mask'],
                            'token_type_ids': batch['token_type_ids'],
                            }
                if self.num_repeats < 1 or cls_idx==0:
                    # out = model.bert(**{'input_ids': batch['input_ids'],
                    #               'attention_mask': batch['attention_mask'],
                    #               'token_type_ids': batch['token_type_ids'],
                    #                })
                    pass
                else:
                    inp_dict['position_ids'] = pos[:len(batch['input_ids'])]
                out = model.bert(**inp_dict)
                cls_output = out.last_hidden_state[:,cls_idx,:].to(device)
                outputs = model.heads[task](outputs = out,
                                            cls_output = cls_output,
                                            return_dict = True,
                                            labels = batch['labels'])

            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])

        result = metric.compute()
        #metric_key = list(result)[0] # we want to return value float (not dict metric -> value)
        metric_key = self.task_to_metric_key(task)
        return result[metric_key]




    def eval_with_prompt(self, dataloader_val, task, metric, cls_idx=0, prefix_len=None, custom_pos_ids=True, pos_ids=None):
        if custom_pos_ids: assert pos_ids != None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.cls_idx_override!=None:
            cls_idx = self.cls_idx_override
        print('Using cls_idx', cls_idx)

        if prefix_len==None:
            prefix_len = self.prefix_len

        model = self.model
        prefix_MLP = None if self.prefix_MLPs==None else self.prefix_MLPs[task]
        if self.same_prompt:
            task_idx = 0
        else:
            task_idx = self.tasks.index(task)
        model.eval().to(device)
        if prefix_MLP!=None:
            prefix_MLP.eval().to(device)

        #metric = load_metric('glue', task)
        if task in glue_datasets:
            metric = load_metric('glue', task)
        else:
            metric = load_metric('accuracy')

        for i, batch in enumerate(tqdm(dataloader_val)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                batch_keys = ['input_ids', 'token_type_ids']
                if custom_pos_ids:
                    batch['position_ids'] = pos_ids.to(device)
                    batch_keys.append('position_ids') # loop through custom position ids
                emb = model.bert.embeddings(**{k: batch[k] for k in batch_keys})

                if prefix_len>0 and prefix_MLP != None:
                    pos1, pos2 = self.max_length + prefix_len*task_idx, self.max_length + prefix_len*(task_idx+1)
                    emb[:, pos1:pos2, :] = prefix_MLP(emb[:, pos1:pos2, :].clone().to(device))
                    #emb[:, :prefix_len, :] = prefix_MLP(emb[:, :prefix_len, :].to(device))

                extended_attention_mask: torch.Tensor = model.bert.get_extended_attention_mask(batch['attention_mask'],
                                                                                                batch['input_ids'].shape,
                                                                                                device=device)
                out = model.bert.encoder(emb, attention_mask=extended_attention_mask)

                cls_output = out.last_hidden_state[:,cls_idx,:].to(device)
                outputs = model.heads[task](outputs = out,
                                            cls_output = cls_output,
                                            return_dict = True,
                                            labels = batch['labels'])

            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])

        res = metric.compute()
        #metric_key = list(res)[0] # we want to return value float (not dict metric -> value)
        metric_key = self.task_to_metric_key(task)
        return res[metric_key]
