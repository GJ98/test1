import collections, json, logging, os
from typing import Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import AutoModelForQuestionAnswering, EvalPrediction, RobertaForQuestionAnswering
from datasets import load_metric, load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn.functional as F


class TfIdfRetriever:

    def __init__(self, config):
        self.config = config
        self.contexts = self.prepare_contexts()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: self.tokenizer.tokenize(x),
            ngram_range=(1, 2)
        )

    def prepare_contexts(self, ):
        with open(self.data.data_path, 'r') as file:
            data = json.load(file)
        return [data[f"{i}"]["text"] for i in range(len(data))]

    def fit(self, ):
        self.vectorizer.fit(self.contexts)
        self.sparse_embedding_matrix = self.vectorizer.transform(self.contexts)

    def add(self, ):
        pass

    def search(self, query, k=1):
        query_vector = self.vectorizer.transform([query])
        similarity = query_vector * self.sparse_embedding_matrix.T
        sorted_result = np.argsort(-similarity.data)
        doc_scores = similarity.data[sorted_result]
        doc_ids = similarity.indices[sorted_result]
        return doc_scores[:k], doc_ids[:k], self.contexts[doc_ids[:k]]


class DenseRetriever:

    def __init__(self, config, q_encoder, p_encoder):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.plm_name)
        self.q_encoder = q_encoder.to(self.config.device)
        self.p_encoder = p_encoder.to(self.config.device)

    def prepare_data(self, ):
        dataset = load_dataset(self.config.data.dataset_name)
        train_dataset = dataset['train'].select(range(100))
        corpus = np.array(list(set([example["context"] for example in train_dataset])))
        p_with_neg = []

        for c in train_dataset['context']:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.config.data.num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        return train_dataset['question'], p_with_neg

    def setup(self, ):
        question, p_with_neg = self.prepare_data()

        q_seqs = self.tokenizer(
            question,
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

        train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"],
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.data.batch_size)

    def fit(self, ):
        trainable_params1 = filter(lambda p: p.requires_grad, self.q_encoder.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, self.p_encoder.parameters())
        self.optimizer = torch.optim.AdamW([
                {"params": trainable_params1},
                {"params": trainable_params2},
            ], **(self.config.optimizer))

        for epoch in range(self.config.train.num_train_epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        self.p_encoder.train()
        self.q_encoder.train()
        targets = torch.zeros(self.config.data.batch_size).long()
        targets = targets.to(self.config.device)

        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f"epoch {epoch} training"):

            p_inputs = {
                "input_ids": batch[0].view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1).to(self.config.device),
                "attention_mask": batch[1].view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1).to(self.config.device),
                "token_type_ids": batch[2].view(self.config.data.batch_size * (self.config.data.num_neg + 1), -1).to(self.config.device)
            }

            q_inputs = {
                "input_ids": batch[3].to(self.config.device),
                "attention_mask": batch[4].to(self.config.device),
                "token_type_ids": batch[5].to(self.config.device)
            }

            self.optimizer.zero_grad()
            p_outputs = self.p_encoder(**p_inputs)
            q_outputs = self.q_encoder(**q_inputs)

            p_outputs = p_outputs.view(self.config.data.batch_size, self.config.data.num_neg+1, -1)
            q_outputs = q_outputs.view(self.config.data.batch_size, 1, -1)

            similarity_scores = torch.bmm(q_outputs, p_outputs.transpose(-2, -1)).squeeze()
            similarity_scores= F.log_softmax(similarity_scores, dim=-1)

            loss = F.nll_loss(similarity_scores, targets)
            loss.backward()
            self.optimizer.step()
            
    def make_dense_embedding_matrix(self, ):
        self.p_encoder.eval()
        dataset = load_dataset(self.config.data.dataset_name)
        eval_dataset = dataset['validation'].select(range(100))
        self.eval_corpus = list(set([example["context"] for example in eval_dataset]))

        p_embs = []
        for p in self.eval_corpus:
            p = self.tokenizer(p, padding='max_length', truncation=True, return_tensors='pt', max_length=self.config.data.max_seq_length).to(self.config.device)
            p_emb = self.p_encoder(**p)
            p_embs.append(p_emb.squeeze())
        
        self.dense_embedding_matrix = torch.stack(p_embs).T
        print(self.dense_embedding_matrix.size())

    def search(self, query, k=1):
        self.q_encoder.eval()

        query_token = self.tokenizer([query], padding='max_length', truncation=True, return_tensors='pt', max_length=self.config.data.max_seq_length).to(self.config.device)
        query_vector = self.q_encoder(**query_token)

        similarity_score = torch.matmul(query_vector, self.dense_embedding_matrix).squeeze()
        sorted_result = torch.argsort(similarity_score, dim=-1, descending=True).squeeze()
        doc_scores = similarity_score[sorted_result]

        return doc_scores[:k], sorted_result[:k], self.eval_corpus[sorted_result[:k]]

class BertEncoder(BertPreTrainedModel):

    def __init__(self,
        config
    ):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()


    def forward(self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
        ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output