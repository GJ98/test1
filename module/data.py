import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset


class MrcDataModule(pl.LightningDataModule):
    ## instance 생성할 때 CFG(baseline_config 세팅) 입력
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
        datasets = load_dataset(self.config.data.dataset_name)
        column_names = datasets['train'].column_names
        if stage == 'fit':
            self.train_examples = datasets['train'].select(range(100))
            self.eval_examples = datasets['validation'].select(range(100))
            #test_examples = datasets['test'].select(range(100))

            self.train_dataset = self.train_examples.map(
                self.prepare_train_features,
                batched=True,
                num_proc=self.config.data.preprocessing_num_workers,
                remove_columns=column_names
            )
            self.eval_dataset = self.eval_examples.map(
                self.prepare_validation_features,
                batched=True,
                num_proc=self.config.data.preprocessing_num_workers,
                remove_columns=column_names
            )
            self.eval_dataset1 = self.eval_dataset.remove_columns("offset_mapping")
            '''
            test_dataset = test_examples.map(
                self.prepare_validation_features,
                num_proc=self.config.data.preprocessing_num_workers,
                remove_columns=column_names
            )
            '''

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.data.batch_size, collate_fn=default_data_collator, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset1, collate_fn=default_data_collator, batch_size=self.config.data.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, collate_fn=default_data_collator, batch_size=self.config.data.batch_size)

    def prepare_train_features(self, examples):
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

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples