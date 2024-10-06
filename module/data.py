import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from utils.data_template import get_dataset_list


class MrcDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.train_examples = None
        self.eval_examples = None
        self.test_examples = None

    def setup(self, stage='fit'):

        dataset_list = get_dataset_list(self.config.data.dataset_name)

        if stage == 'fit':
            datasets = DatasetDict()
            for split in ['train', 'validation']:
                datasets[split] = concatenate_datasets([ds[split] for ds in dataset_list])

            self.train_examples = datasets['train'].select(range(100))
            self.eval_examples = datasets['validation'].select(range(100))
            self.train_dataset = self.get_dataset(self.train_examples, self.prepare_train_features)
            self.eval_dataset = self.get_dataset(self.eval_examples, self.prepare_validation_features)
            print(self.train_dataset)
            print(self.eval_dataset)

        if stage == "test":
            datasets = DatasetDict()
            self.test_examples = concatenate_datasets([ds["test"] for ds in dataset_list]).select(range(100))
            self.test_dataset = self.get_dataset(self.test_examples, self.prepare_validation_features)
            print(self.test_dataset)

    def get_dataset(self, examples, preprocess_func):
        return examples.map(
            preprocess_func,
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers,
            remove_columns=examples.column_names
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, \
                    batch_size=self.config.data.batch_size, collate_fn=default_data_collator, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset.remove_columns('offset_mapping'), \
                    collate_fn=default_data_collator, batch_size=self.config.data.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset.remove_columns('offset_mapping'), \
                    collate_fn=default_data_collator, batch_size=self.config.data.batch_size)

    def prepare_train_features(self, examples):
        """
        Args:
            examples: {'id', 'question', 'context', 'answers': {'answer_start', 'text'}}
        Returns:
            tokenized_examples: {'input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'}
        """
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.config.data.max_seq_length,
            stride=self.config.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            #cls_index = input_ids.index(self.tokenizer.sep_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            #answers = examples["answers"]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        """
        Args:
            examples: {'id', 'question', 'context', 'answers': {'answer_start', 'text'}}
        Returns:
            tokenized_examples: {'input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id'}
        """
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.config.data.max_seq_length,
            stride=self.config.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            #tokenized_examples["example_id"].append(examples["id"])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

class RetrievalDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def setup(self, stage='fit'):

        dataset_list = get_dataset_list(self.config.data.dataset_name)

        if stage == "fit":
            datasets = DatasetDict()
            for split in ['train', 'validation']:
                datasets[split] = concatenate_datasets([ds[split] for ds in dataset_list])

            train_dataset = datasets['train'].select(range(100))
            eval_dataset = datasets['validation'].select(range(100))

            self.train_dataset = self.preprocessing(train_dataset)
            self.eval_dataset = self.preprocessing(eval_dataset)

        if stage == "test":
            test_dataset = concatenate_datasets([ds['test'] for ds in dataset_list]).select(range(100))
            self.test_dataset = self.preprocessing(test_dataset)

    def preprocessing(self, dataset):

        corpus = np.array(list(set([example["context"] for example in dataset])))
        p_with_neg = []

        for context in dataset['context']:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.config.data.num_neg)

                if not context in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    p_with_neg.append(context)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = self.tokenizer(
            dataset["question"],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=self.config.data.max_seq_length
        )

        p_seqs = self.tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.config.data.max_seq_length
        )

        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, self.config.data.num_neg + 1, self.config.data.max_seq_length)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, self.config.data.num_neg + 1, self.config.data.max_seq_length)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, self.config.data.num_neg + 1, self.config.data.max_seq_length)

        return TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"],
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, \
                    batch_size=self.config.data.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.config.data.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.data.batch_size)