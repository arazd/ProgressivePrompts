from datasets import load_dataset
import datasets
import numpy as np
import pandas as pd
import torch

class Dataset:
    def __init__(self, tokenizer, task, idbr_preprocessing=False):
        self.task = task
        self.tokenizer = tokenizer
        self.idbr_preprocessing = idbr_preprocessing # do idbr-style sentence preprocessing (split sentence in 2 parts and add sep token)

        self.task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            #"qnli": ("question", "sentence"),
            "qnli": ("text1", "text2"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),

            "scicite": ("sectionName", "string"),
            "imdb": ("text", None),

            "cb": ("premise", "hypothesis"),
            "boolq": ("passage", "question"),
            "copa": ('choice1', 'choice2', 'premise', 'question'),
            "wic": ("start1", "end1", "sentence1", "start2", "end2", "sentence2", "word"),
            "wsc": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
            "multirc": ("question", "answer", "paragraph"),

            "ag_news": ("text", None),
            "yelp_review_full": ("text", None),
            "yahoo_answers_topics": ("question_content", "best_answer"),
            "dbpedia_14": ("title", "content"),

            "ag": ("content", None),
            "yelp": ("content", None),
            "yahoo": ("content", None),
            "dbpedia": ("content", None),
            "amazon": ("content", None),
        }

        # self.sentence1_key, self.sentence2_key = self.task_to_keys[task]

    def preprocess_function(self, examples, max_length=100):
        #sentence1_key, sentence2_key = self.task_to_keys[self.task]
        sentence_keys = self.task_to_keys[self.task]

        if self.idbr_preprocessing and sentence_keys[1]==None:
            sentence1_key, sentence2_key = self.task_to_keys[self.task]
            tokenized_res = self.tokenizer.tokenize(examples[sentence1_key])
            ids = self.tokenizer.convert_tokens_to_ids(tokenized_res)[:max_length-3]
            len1 = len(ids) // 2
            x = [101] + ids[:len1] + [102] + ids[len1:] + [102]
            mask = [1] * len(x)

            padding = [0] * (max_length - len(x))
            x += padding
            mask += padding

            d = {'input_ids': x, 'attention_mask': mask, 'token_type_ids': [0]*max_length}

            assert len(x) == max_length
            assert len(mask) == max_length

            return d

        if self.task == "yahoo_answers_topics":
            return self.tokenizer(examples["question_title"] + '[SEP]' + examples["question_content"] + '[SEP]' + \
                                  examples["best_answer"],
                                  truncation=True,
                                  padding='max_length',
                                  max_length=max_length)

        if self.task in ["copa", "wic", "wsc", "multirc"]:
            text = ('[SEP]').join([str(examples[x]) for x in self.task_to_keys[self.task]])
            return self.tokenizer(text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=max_length)

        # for all other tasks we have 2 sentence keys
        sentence1_key, sentence2_key = self.task_to_keys[self.task]
        if sentence2_key is None:
            return self.tokenizer(examples[sentence1_key],
                                  truncation=True,
                                  #padding=False,
                                  padding='max_length',
                                  max_length=max_length)
        return self.tokenizer(examples[sentence1_key], examples[sentence2_key],
                              truncation=True,
                              #padding=False,
                              padding='max_length',
                              max_length=max_length)



    def encode_with_repeats(self, examples, prefix_tokens_list=[], repeats=0, prefix_len=0, max_length=100, do_repeats=False):
        # max length defines input length not counting prompt (in case of repeats, each repeat will be of max length)
        #text = examples['sentence']
        #inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
        if prefix_len==0:
            inputs = self.preprocess_function(examples, max_length=max_length) # add 1 since we will remove CLS token

        if prefix_len>0:
            soft_prompt = []
            inputs = self.preprocess_function(examples, max_length=max_length+1) # add 1 since we will remove CLS token
            for j in range(prefix_len):
                tokenized = self.tokenizer.tokenize(prefix_tokens_list[j])
                #cls1_id = self.tokenizer.convert_tokens_to_ids(tokenized)[0]
                soft_prompt.append(self.tokenizer.convert_tokens_to_ids(tokenized)[0])

            if repeats == 0 and not do_repeats:
                #print('We got soft prompt: ', soft_prompt)
                L = max_length + prefix_len # total len
                # for n in range(len(inputs['input_ids'])):
                #     inputs['input_ids'][n] = inputs['input_ids'][n][1:][:max_length] + soft_prompt # ignore CLS token since we have one in prefix
                #     for key in ['token_type_ids', 'attention_mask']:
                #         inputs[key][n] = inputs[key][n][1:][:max_length] + inputs[key][n][0:1]*prefix_len

                # if batched == False
                inputs['input_ids'] = inputs['input_ids'][1:][:max_length] + soft_prompt # ignore CLS token since we have one in prefix
                for key in ['token_type_ids', 'attention_mask']:
                    inputs[key] = inputs[key][1:][:max_length] + inputs[key][0:1]*prefix_len

        if repeats>0 or do_repeats: # if we have 1+ repeats or current sentence is the 0th repeat (should be formatted accordingly)
            assert prefix_len>0
            # cls_j_id = {}
            # for j in range(repeats):
            #     tokenized = self.tokenizer.tokenize(prefix_tokens_list[j])
            #     cls_j_id[j] = self.tokenizer.convert_tokens_to_ids(tokenized)[0]

            #for n in range(len(inputs['input_ids'])): # we are not currently using batched version
            repeat = inputs['input_ids'].copy()[1:][:max_length]
            repeat_full = []
            single_prefix_len = prefix_len//(repeats+1)
            for j in range(repeats+1):
                repeat_full += soft_prompt[single_prefix_len*j : single_prefix_len*(j+1)] + repeat.copy()
            inputs['input_ids'] = repeat_full

            for key in ['token_type_ids', 'attention_mask']:
                repeat_block = inputs[key][0:1]*single_prefix_len + inputs[key][1:][:max_length]
                inputs[key] = repeat_block*(repeats+1)

        return inputs


    def prepare_dataset(self,
                        dataset,
                        prefix_tokens_list=[],
                        prefix_len=0,
                        repeats=0,
                        batch_size=8,
                        max_length=100,
                        task=None,
                        label_offset=0,
                        do_repeats=False): # input = dataset loaded from hugging face
        encoded_dataset = dataset.map(lambda x: self.encode_with_repeats(x,
                                                                         repeats=repeats,
                                                                         prefix_tokens_list=prefix_tokens_list,
                                                                         prefix_len=prefix_len,
                                                                         max_length=max_length,
                                                                         do_repeats=do_repeats), batched=False)
        if task==None:
            task = self.task
        label_key = 'label' if 'yahoo_' not in task else 'topic'

        if label_offset==0: # no change to the original label
            dataset2 = encoded_dataset.map(lambda examples: {'labels': examples[label_key]}, batched=True)
        else:  # adding offset (for one head training)
            dataset2 = encoded_dataset.map(lambda examples: {'labels': examples[label_key] + label_offset})
        dataset2.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        dataloader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

        return dataloader


    def select_subset_ds(self, ds, k=2000, task=None, seed=0):
        if task==None:
            task = self.task

        if self.task=='stsb': # stsb has continuos labels
            idx_total = np.random.choice(np.arange(ds.shape[0]), min(k, ds.shape[0]), replace=False)
        else:
            label_key = 'label' if 'yahoo_' not in task else 'topic'
            N = len(ds[label_key])
            idx_total = np.array([], dtype='int64')

            for l in set(ds[label_key]):
                idx = np.where(np.array(ds[label_key]) == l)[0]
                idx_total = np.concatenate([idx_total, np.random.choice(idx, min(k, idx.shape[0]), replace=False)])

        np.random.seed(seed)
        np.random.shuffle(idx_total)
        return ds.select(idx_total)


    def get_dataset(self,
                    #task='cola',
                    benchmark=None,
                    prefix_tokens_list=[],
                    prefix_len=0,
                    split='train',
                    repeats=0,
                    batch_size=8,
                    select_k_per_class=-1,
                    max_length=100,
                    return_test_subset=False,
                    label_offset=0,
                    seed=42,
                    do_repeats=False):
        #dataset = load_dataset('glue', 'cola', split='train')
        task = self.task
        # if task == 'amazon':
        #     # amazon reviews (with 5 starts) is only available from original paper's google drive
        #     df = pd.read_csv('downloaded_data/amazon_review_full_csv/'+split+'.csv', header=None)
        #     df = df.rename(columns={0: "label", 1: "title", 2: "content"})
        #     df['label'] = df['label'] - 1
        #     dataset = datasets.Dataset.from_pandas(df)

        if task in ['cola', 'rte', 'mrpc', 'cb', 'copa', 'wsc'] and select_k_per_class>250:
            select_k_per_class = -1 # too small datasets for selection

        if task in ['ag', 'yahoo', 'yelp', 'amazon', 'dbpedia']:
            df = pd.read_csv('../datasets/src/data/'+task+'/'+split+'.csv', header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df['label'] = df['label'] - 1
            dataset = datasets.Dataset.from_pandas(df)

        elif task == 'mnli':
            dataset = load_dataset('LysandreJik/glue-mnli-train', split=split)
        elif task == 'qnli':
            dataset = load_dataset('SetFit/qnli', split=split)
        elif task == 'stsb':
            dataset = load_dataset('stsb_multi_mt', name='en', split=split if split=='train' else 'dev')
        else:
            if benchmark != None:
                dataset = load_dataset(benchmark, task, split=split)
            else:
                dataset = load_dataset(task, split=split)

        if self.task == "yahoo_answers_topics":
        # for yahoo dataset we need to filter out empty rows (no question)
            if split=='train':
                good_id = np.load('good_id_yahoo_train.npy')
                dataset = dataset.select(good_id)
            elif split=='test':
                good_id = np.load('good_id_yahoo_test.npy')
                dataset = dataset.select(good_id)

        dataset = dataset.shuffle(seed=seed)

        if select_k_per_class != -1 and task not in ['copa', 'cb', 'wic']:
            k = select_k_per_class
            if return_test_subset:
                k *= 2
            dataset = self.select_subset_ds(dataset, k=k)

        if not return_test_subset:
            # returning one dataset
            dataset_final = self.prepare_dataset(dataset,
                                                 repeats=repeats,
                                                 batch_size=batch_size,
                                                 max_length=max_length,
                                                 prefix_tokens_list=prefix_tokens_list,
                                                 prefix_len=prefix_len,
                                                 label_offset=label_offset,
                                                 do_repeats=do_repeats)
            return dataset_final

        else:
            # splitting current dataset into 2: val and test
            N = len(dataset)
            dataset_val = dataset.select(np.arange(0, N//2))
            dataset_test = dataset.select(np.arange(N//2, N))

            dataset_final_val = self.prepare_dataset(dataset_val, repeats=repeats, do_repeats=do_repeats,
                                                     batch_size=batch_size, max_length=max_length,
                                                     prefix_tokens_list=prefix_tokens_list, prefix_len=prefix_len,
                                                     label_offset=label_offset)
            dataset_final_test = self.prepare_dataset(dataset_test, repeats=repeats, do_repeats=do_repeats,
                                                      batch_size=batch_size, max_length=max_length,
                                                      prefix_tokens_list=prefix_tokens_list, prefix_len=prefix_len,
                                                      label_offset=label_offset)
            return dataset_final_val, dataset_final_test
