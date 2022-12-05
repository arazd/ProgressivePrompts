import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import t5_model, t5_dataset
from copy import deepcopy
from transformers import AdamW


class ResMLP(torch.nn.Module):
    def __init__(self, bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=512,
                 residual=True,
                 dropout=0.0,
                 #layer_norm=True
                 ):
        super().__init__()
        if module_type=='MLP1':
            if dropout>0:
                self.module = nn.Sequential(
                    nn.Linear(emb_dimension, bottleneck_size),
                    nn.ReLU(),
                    #nn.Tanh(),
                    nn.Linear(bottleneck_size, emb_dimension),
                    #nn.LayerNorm(emb_dimension),
                    nn.Dropout(dropout)
                )
            else:
                self.module = nn.Sequential(
                    nn.Linear(emb_dimension, bottleneck_size),
                    #nn.ReLU(),
                    nn.Tanh(),
                    nn.Linear(bottleneck_size, emb_dimension),
                )

        elif module_type=='MLP2':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_size // 2, emb_dimension),
                #nn.LayerNorm(emb_dimension),
            )

        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)



def train_step_lester(trainer, batch, prefix_len, embed_prompt=False):
    model = trainer.model
    if embed_prompt:
        mlp = model.mlp
    tokenizer = trainer.tokenizer

    batch = {k: batch[k].to(trainer.device) for k in batch}
    lm_labels = batch["target_ids"]
    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

    inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])

    k = inputs_embeds.shape[0]
    if embed_prompt:
        prompt = mlp(model.prompt)
    else:
        prompt = model.prompt

    inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                  inputs_embeds], axis=1)[:,:512]

    source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,prefix_len),
                                         batch["source_mask"]), axis=1)[:,:512]

    encoder_outputs = model.encoder(
                            #input_ids=batch["source_ids"],
                            attention_mask=source_mask_updated, #batch["source_mask"],
                            #labels=lm_labels,
                            #decoder_attention_mask=batch['target_mask']
                            #input_ids=input_ids,
                            #attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds,
                            head_mask=None, #head_mask,
                            output_attentions=None, #output_attentions,
                            output_hidden_states=None, #output_hidden_states,
                            return_dict=None, #return_dict,
                        )

    outputs = model(
        input_ids=batch["source_ids"],
        attention_mask=source_mask_updated, #batch["source_mask"],
        labels=lm_labels,
        decoder_attention_mask=batch['target_mask'],
        encoder_outputs=encoder_outputs,
    )
    loss = outputs[0]

    return loss



def validate_lester(trainer, dataloader_val, task, embed_prompt, prefix_len,
                    class_keys=['equivalent', 'different'],
                    max_length=2, print_outputs=False):
    model = trainer.model
    if embed_prompt:
        mlp = model.mlp
        prompt = mlp(model.prompt)
    else:
        prompt = model.prompt

    tokenizer = trainer.tokenizer
    model.eval()

    corr, total = 0, 0
    loss_total = []
    # try:
    #     metric = datasets.load_metric('glue', task)
    # except:
    #     metric = datasets.load_metric('accuracy')

    for i, batch in enumerate(tqdm(dataloader_val)):
        batch = {k:batch[k].to(trainer.device) for k in batch}

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"]).to(trainer.device)
        k = inputs_embeds.shape[0]

        inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                      inputs_embeds], axis=1)[:,:512]

        source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,prefix_len),
                                             batch["source_mask"]), axis=1)[:,:512]


        encoder_outputs = model.encoder(
                                #input_ids=batch["source_ids"],
                                #attention_mask=batch["source_mask"],
                                attention_mask=source_mask_updated,
                                #labels=lm_labels,
                                #decoder_attention_mask=batch['target_mask']
                                #input_ids=input_ids,
                                #attention_mask=attention_mask,
                                inputs_embeds=inputs_embeds,
                                head_mask=None, #head_mask,
                                output_attentions=None, #output_attentions,
                                output_hidden_states=None, #output_hidden_states,
                                return_dict=None, #return_dict,
                            )

        outs = model.generate(
            input_ids=batch["source_ids"],
            #attention_mask=batch["source_mask"],
            attention_mask=source_mask_updated,
            #labels=lm_labels,
            #decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
            max_length=max_length,
        )
        dec = [tokenizer.decode(ids) for ids in outs]
        texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
        targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

        #print(dec, texts, targets)
        corr += np.sum([trainer.process_str(x)==trainer.process_str(y) for x,y in zip(dec, targets)])
        total += batch['source_ids'].shape[0]

        if i<10 and print_outputs:
            print(dec)
            print(targets)

        # CHANGE FOR MULTI CLASS!!!
        # metric.add_batch(predictions=[1 if class_keys[1] in x else 0 for x in dec],
        #                  references=[1 if class_keys[1] in x else 0 for x in targets])

        # computing loss
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == trainer.tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["source_ids"],
            #attention_mask=batch["source_mask"],
            attention_mask=source_mask_updated,
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        loss = outputs[0].detach().cpu().numpy()
        loss_total.append(loss)

    return corr/total, np.mean(loss_total)


def train(TrainerT5,
          task,
          dataloader_train,
          dataloader_val,
          embed_prompt,
          class_keys,
          target_len,
          prefix_len,
          epochs=40,
          save_path=None):

    print('task = ', task)
    print("Using MLP? ", embed_prompt)
    model = TrainerT5.model
    model.to('cuda')

    #embed_prompt = True
    results_dict = {}
    results_dict['val'] = {'acc': [], 'loss': []}
    results_dict['train'] = {'acc': [], 'loss': []}

    for epoch in range(epochs):

        model.train()
        #mlp.train()

        for i, batch in enumerate(tqdm(dataloader_train)):
            batch = {k:batch[k].to('cuda') for k in batch}
            #loss = train_step_lester(TrainerT5, batch, TrainerT5.model.prompt, embed_prompt=embed_prompt)
            loss = train_step_lester(TrainerT5, batch, prefix_len, embed_prompt=embed_prompt)
            loss.backward()

            TrainerT5.optimizer.step()
            TrainerT5.optimizer.zero_grad()

        for dataloader, name in zip([dataloader_val, dataloader_train],
                                     ['val', 'train']):
        #for dataloader, name in zip([dataloader_val],
        #                            ['val']):

            acc, loss = validate_lester(TrainerT5, dataloader_val, task,
                                        embed_prompt, prefix_len,
                                        class_keys=class_keys,
                                        max_length=target_len, print_outputs=True)
            results_dict[name]['acc'].append(acc)
            results_dict[name]['loss'].append(loss)
            print(epoch, name, '->', acc, loss)
            #print('train acc ->', train_acc, train_f1)

        if save_path!=None and epoch%5==0:
            np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)
    return results_dict




def get_prompt(trainer, prompt_len):
    model = trainer.model
    N = model.encoder.embed_tokens.weight.shape[0]
    prompt_weigths = []

    for i in range(prompt_len):
        with torch.no_grad():
            j = np.random.randint(N)
            #j = 21
            w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
            prompt_weigths.append(w)
    prompt_weigths = np.array(prompt_weigths)
    return prompt_weigths



def main(args):

    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    TrainerT5= t5_model.PromptModelT5(model_name=args.model_name,
                                      prefix_len=0,
                                      freeze_weights=args.freeze_weights==1,
                                      freeze_except='xxxshared', # freeze all weights
                                      lr=args.lr,
                                      weight_decay=0.00,
                                      prompt_name='PRE',
                                      prefix_MLP='None', # using custom prefix MLP
                                      #mlp_bottleneck=args.mlp_bottleneck,
                                      #weight_decay_mlp=0.0,
                                      #mlp_lr=args.lr_mlp,
                                      #mlp_layer_norm=args.mlp_layer_norm==1,
                                      early_stopping=False,
                                      #opt=args.optimizer,
                                     )

    prompt_weigths = get_prompt(TrainerT5, prompt_len=args.prefix_len)
    TrainerT5.model.prompt = nn.Parameter(torch.tensor(prompt_weigths, requires_grad=True))
    print('created prompt: ', prompt_weigths.shape)

    if args.prefix_MLP != 'None':
        # adding MLP reparametrization for prompt
        print('Using MLP')
        TrainerT5.model.mlp = ResMLP(bottleneck_size=args.mlp_bottleneck,
                                     residual=args.residual_mlp==1,
                                     module_type=args.prefix_MLP,
                                     emb_dimension=prompt_weigths.shape[1],
                                     dropout=args.mlp_dropout,
                                     #layer_norm=False
                                     )

    lr_mlp = args.lr_mlp if args.lr_mlp!=-1 else args.lr
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in TrainerT5.model.named_parameters() if n=='prompt'],
            "weight_decay": 1e-5,
            "lr": args.lr,
        },

        {
            "params": [p for n, p in TrainerT5.model.named_parameters() if 'mlp' in n],
            "weight_decay": 1e-5,
            "lr": lr_mlp,
        },

    ]

    TrainerT5.optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    #TrainerT5.optimizer = Adafactor(optimizer_grouped_parameters)# eps=1e-8)
    #TrainerT5.optimizer

    task = args.task #'mrpc'
    target_len = args.target_len #2
    if task=='rte' or task=='mrpc': target_len=5

    ds2 = t5_dataset.T5Dataset(TrainerT5.tokenizer, task)
    dataloader_train = ds2.get_final_ds(task, 'train', batch_size=args.batch_size, k=args.select_k_per_class,
                                        target_len=target_len, prefix_list=[])

    k_val = -1 if (args.select_k_per_class==-1 or task in ['mrpc', 'rte']) else int(0.2*args.select_k_per_class)
    dataloader_val = ds2.get_final_ds(task, 'validation',
                                      batch_size=args.batch_size, k=k_val, return_test=False,
                                      target_len=target_len, prefix_list=[])

    class_keys = ds2.task_to_labels[task]
    results_dict = train(TrainerT5,
                         task,
                         dataloader_train,
                         dataloader_val,
                         embed_prompt=args.prefix_MLP != 'None',
                         class_keys=class_keys,
                         target_len=target_len,
                         prefix_len=args.prefix_len,
                         epochs=args.epochs,
                         save_path=save_path)

    if args.early_stopping==1:
        TrainerT5.load_best_model() # for early stopping

    # test_acc, test_f1 = validate(TrainerT5, dataloader_test, task, ds.task_to_labels[task], target_len)
    # results_dict['test'] = {}
    # results_dict['test']['acc_direct'] = test_acc
    # for key in test_f1:
    #     results_dict['test'][key] = test_f1[key]
    np.save(os.path.join(save_path, 'results_dict.npy'), results_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features (should not be changed)',
        default='/data/home/arazdai/T5_prompts/results/'
    )

    parser.add_argument(
        '--save_name',
        type=str,
        help='folder name to save',
        required=True
    )

    parser.add_argument(
        '--task',
        type=str,
        help='task to train t5 (e.g. mrpc, cola, rte)',
        required=True
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='t5 model type',
        default='t5-small'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size',
        default=8
    )


    # parser.add_argument(
    #     '--optimizer',
    #     type=str,
    #     help='Which optimizer to use? (AdamW, LAMB etc.)',
    #     default='AdamW'
    # )

    parser.add_argument(
        '--select_k_per_class',
        type=int,
        help='Select k instances per class (default -1 = select all)',
        default=-1
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs',
        default=50
    )

    parser.add_argument(
        '--target_len',
        type=int,
        help='maximum length of the output (in tokens)',
        default=2
    )

    parser.add_argument(
        '--prefix_len',
        type=int,
        help='prompt length (in tokens)',
        default=50
    )

    parser.add_argument(
        '--early_stopping',
        type=int,
        help='Perform early stopping (1 - True, 0 - False)',
        default=1
    )

    parser.add_argument(
        '--freeze_weights',
        type=int,
        help='Whether to freeze model weigts',
        default=0
    )

    # parser.add_argument(
    #     '--freeze_except',
    #     type=str,
    #     help='If freeze_weights==1, freeze all weights except those that contain this keyword',
    #     default='shared' # shared stands for wte
    # )




    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=0.01
    )

    parser.add_argument(
        '--mlp_dropout',
        type=float,
        help='Dropout rate of MLP (if 0 then no dropout)',
        default=0.0
    )

    parser.add_argument(
        '--lr_mlp',
        type=float,
        help='Learning rate of MLP (if -1 then use the same LR as prompt)',
        default=-1
    )

    parser.add_argument(
        '--prefix_MLP',
        type=str,
        help='Whether to do embeddings reparametrization with prefix MLP',
        default='None'
    )

    parser.add_argument(
        '--mlp_bottleneck',
        type=int,
        help='MLP bottleneck size',
        default=1000
    )

    parser.add_argument(
        '--residual_mlp',
        type=int,
        help='Whether to use skip connection in MLP',
        default=1
    )


    # parser.add_argument(
    #     '--mlp_layer_norm',
    #     type=int,
    #     help='Whether to use MLP layer norm (1 - use, 0 - not use)',
    #     default=0
    # )


    main(parser.parse_args())
