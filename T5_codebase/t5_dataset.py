import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets


class T5Dataset:
    def __init__(self, tokenizer, task):
        """Dataset class for T5 model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): T5 model tokenizer to use.
        """
        
        self.tokenizer = tokenizer
        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'cb', 'record', 'multirc', 'rte_superglue', 'wsc_bool']
        
        # Column keys used in the dataset
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

            "boolq": ("passage", "question"),
            "copa": ('choice1', 'choice2', 'premise', 'question'),
            "wic": ("start1", "end1", "sentence1", "start2", "end2", "sentence2", "word"),
            "wsc": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
            "wsc_bool": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
            "cb": ("premise", "hypothesis"),
            "record": ("passage", "query", "entities"),
            "multirc": ("question", "answer", "paragraph"),
            "rte_superglue": ("premise", "hypothesis"),

            "scicite": ("sectionName", "string"),
            "imdb": ("text", None),

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
        
        # Label text for T5 tasks
        # (T5 has text-to-text format for text and labels)
        self.task_to_labels = {
            "cola": ("not_acceptable", "acceptable"),
            "mnli": ("entailment", "neutral", "contradiction"),
            "mnli-mm": (),
            "mrpc": ("not_equivalent", "equivalent"),
            "qnli": ("entailment", "not_entailment"),
            "qqp": ("not_duplicate", "duplicate"),
            "rte": ("entailment", "not_entailment"),
            "sst2": ("negative", "positive"),
            "stsb": (),
            "wnli": (),

            "boolq": ("false", "true"),
            "copa": ("false", "true"),
            "wic": ("false", "true"),
            "wsc_bool": ("false", "true"),
            "cb": ("entailment", "contradiction", "neutral"),
            "multirc": ("false", "true"),
            "rte_superglue": ("entailment", "not_entailment"),

            "scicite": (),
            "imdb": ("negative", "positive"),

            "ag_news": ("world", "sports", "business", "science"),
            "yelp_review_full": ("terrible", "bad", "middle", "good", "wonderful"),
            "yahoo_answers_topics": ("society and culture", "science", "health", "education and reference",
                                     "computers and internet", "sports", "business", "entertainment and music",
                                     "family and relationships", "politics and government"),
            "dbpedia_14": ("company", "educationalinstitution", "artist", "athlete", "officeholder",
                           "meanoftransportation", "building", "naturalplace", "village", "animal",
                           "plant", "album", "film", "writtenwork"),

            "ag": ("world", "sports", "business", "science"),
            "yelp": ("terrible", "bad", "middle", "good", "wonderful"),
            "yahoo": ("society and culture", "science", "health", "education and reference",
                      "computers and internet", "sports", "business", "entertainment and music",
                      "family and relationships", "politics and government"),
            "dbpedia": ("company", "educationalinstitution", "artist", "athlete", "officeholder",
                        "meanoftransportation", "building", "naturalplace", "village", "animal",
                        "plant", "album", "film", "writtenwork"),
            "amazon": ("terrible", "bad", "middle", "good", "wonderful"),
        }

        self.task = task
        self.label_key = 'label'
        if 'yahoo_' in task: self.label_key = 'topic'
        if 'stsb' in task: self.label_key = 'similarity_score'
        if task=='record': self.label_key = 'answers'

    
    # Helper function to save idx of multirc questions (needed later for test metric computation)
    def save_multirc_questions_idx(self, val_ds):
        idx = []
        i = 0
        x_prev, y_prev= val_ds['paragraph'][0], val_ds['question'][0]

        for x,y in zip(val_ds['paragraph'], val_ds['question']):
            if x_prev!=x or y_prev!=y:
                i += 1
            x_prev = x
            y_prev = y
            idx.append(i)
        self.multirc_idx = np.array(idx)

    
    # Helper function to select a subset of k samples per class in a dataset
    def select_subset_ds(self, ds, k=2000, seed=0):
        if self.task in ['stsb', 'record', 'wsc']: # non-discrete labels
            idx_total = np.random.choice(np.arange(ds.shape[0]), min(k,ds.shape[0]), replace=False)

        else:
            label_key = self.label_key
            N = len(ds[label_key])
            idx_total = np.array([], dtype='int64')

            for l in set(ds[label_key]):
                idx = np.where(np.array(ds[label_key]) == l)[0]
                idx_total = np.concatenate([idx_total, # we cannot take more samples than there are available
                                            np.random.choice(idx, min(k, idx.shape[0]), replace=False)])

        np.random.seed(seed)
        np.random.shuffle(idx_total)
        return ds.select(idx_total)

    
    # WSC task function to preprocess raw input & label text into tokenized dictionary
    def process_wsc(self, wsc_row):
        text_proc = wsc_row['text'].split(' ')
        #text_proc[wsc_row['span1_index']] = '*' + text_proc[wsc_row['span1_index']] +'*'
        target = text_proc[wsc_row['span1_index']]
        text_proc[wsc_row['span2_index']] = '*' + text_proc[wsc_row['span2_index']] + '*'
        text_proc = (' ').join(text_proc)
        return text_proc, target

    
    # Function to preprocess raw input & label text into tokenized dictionary
    def preprocess_function(self, examples, task,
                            max_length=512, max_length_target=2,
                            prefix_list=[]):
        tokenizer = self.tokenizer
        keys = self.task_to_keys[task]
        label_key = self.label_key

        if keys[1]!=None:
            if task=='record':
                text = 'passage : ' + str(examples['passage']) + ' query: ' + str(examples['query']) + ' entities: ' + ('; ').join((examples['entities']))
            elif task=='wsc':
                text, target = self.process_wsc(examples)
            else:
                text = ''
                for key in keys:
                    text += key + ': ' + str(examples[key]) + ' '
        else:
            text = examples[keys[0]]

        if len(prefix_list)>0:
            text = (' ').join(prefix_list) + ' ' + text
        source = tokenizer(text.strip()+' </s>',
                          truncation=True,
                          #padding=False,
                          padding='max_length',
                          max_length=max_length)

        if task=='stsb':
            target = str(examples[label_key])[:3]
        elif task=='record':
            target = '; '.join(examples[label_key])
        elif task=='wsc':
            pass # already obtained target
        else:
            target = self.task_to_labels[task][examples[label_key]]
        target += ' </s>'
        target = tokenizer(
                  target, max_length=max_length_target, pad_to_max_length=True, #return_tensors="pt"
                )

        dict_final = {"source_ids": source['input_ids'],
                      "source_mask": source['attention_mask'],
                      "target_ids": target['input_ids'],
                      "target_mask": target['attention_mask']}
        return dict_final


    
    def get_final_ds(self, 
                     task, 
                     split,
                     batch_size,
                     k=-1,
                     seed=0,
                     return_test=False,
                     target_len=2,
                     max_length=512,
                     prefix_list=[]):
        """Function that returns final T5 dataloader.
        Args:
            task (str): Name of the downstream task.
            split (str): Which data split to use (train/validation/test).
            batch_size (int): Batch size to use in the dataloader.
            k (int, optional): Number of samples to use for each class. Defaults to -1, not sub-sample the data.
            seed (int, optional): Seed used for random shuffle. Defaults to 0.
            return_test (bool, optional): Whether to create a test split. 
                When True, two Dataloaders are returned. Defaults to False.
            target_len (int, optional): Length of the model output (in tokens). Defaults to 2.
            max_length (int, optional): Length of the model input (in tokens). Defaults to 512.
            prefix_list (List[str], optional): List of prompt virtual tokens to pre-pend to the input. 
                We do not encode soft prompt as extra virtual tokens in the latest implementation.
                Defaults to [], empty list.
            
        Returns:
            Dataloader: Torch Dataloader with preprocessed input text & label.
        """
        
        if task in ['amazon']: # amazon not available with hugging face
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
            if task not in self.glue_datasets and task not in self.superglue_datasets:
                dataset = load_dataset(task, split=split)
            else:
                benchmark = 'glue' if task not in self.superglue_datasets else 'super_glue'
                dataset = load_dataset(benchmark,
                                       task.replace('_superglue', '').replace('_bool', ''),
                                       split=split)

        # For yahoo dataset we need to filter out empty rows 
        # (i.e. where "question" field is empty)
        if self.task == "yahoo_answers_topics":
            if split=='train':
                good_id = np.load('good_id_yahoo_train.npy')
                dataset = dataset.select(good_id)
            elif split=='test':
                good_id = np.load('good_id_yahoo_test.npy')
                dataset = dataset.select(good_id)
        
        # Using Lester et al. setting for WSC task, e.g.
        # using only positive samples (for output generation)
        if self.task == 'wsc': 
            idx = np.where(np.array(dataset['label']) == 1)[0]
            dataset = dataset.select(idx)
        
        # Selecting k subset of the samples (if requested)
        if k!=-1:
            dataset = self.select_subset_ds(dataset, k=k)

        if k==-1 and split!='train' and self.task=='multirc':
            # we do not shuffle full validation set of multirc
            # but we save idx of the same questions
            # which are used for multirc test metric computation
            self.save_multirc_questions_idx(dataset)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        # Returning the selected data split (train/val/test)
        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len,
                                                                            prefix_list=prefix_list),
                                          batched=False)
            encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
            dataloader = DataLoader(encoded_dataset, batch_size=batch_size)

            return dataloader
        
        # Creating an extra test set from the selected data split
        else:
            N = len(dataset)
            dataset_val = dataset.select(np.arange(0, N//2))
            dataset_test = dataset.select(np.arange(N//2, N))

            dataloaders_val_test = []
            for dataset in [dataset_val, dataset_test]:
                encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                                 max_length=max_length,
                                                                                 max_length_target=target_len,
                                                                                 prefix_list=prefix_list),
                                              batched=False)
                encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                                  'target_ids', 'target_mask'])
                dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
                dataloaders_val_test.append(dataloader)

            return dataloaders_val_test
