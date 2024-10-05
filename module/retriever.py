import collections, json, logging, os
from typing import Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import AutoModelForQuestionAnswering, EvalPrediction, RobertaForQuestionAnswering
from datasets import load_metric, load_dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn.functional as F
import torchmetrics
import module.loss as module_loss
from utils.common import init_obj

from utils.data_template import get_dataset_list
from evaluate import load
from utils.common import init_obj
import module.metric as module_metric

class TfIdfRetriever:

    def __init__(self, config):
        self.config = config
        self.contexts = self.prepare_contexts()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.retriever_plm_name)
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: self.tokenizer.tokenize(x),
            ngram_range=(1, 2)
        )
        self.sparse_embedding_matrix = None

    def prepare_contexts(self, ):
        with open(self.data.data_path, 'r') as file:
            data = json.load(file)
        return [data[f"{i}"]["text"] for i in range(len(data))]

    def fit(self, ):
        self.vectorizer.fit(self.contexts)
        self.sparse_embedding_matrix = self.vectorizer.transform(self.contexts)

    def create_embedding_vector(self, ):
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


class DenseRetriever(pl.LightningModule):

    def __init__(self, config, q_encoder, p_encoder):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.encoder_plm_name)
        self.q_encoder = q_encoder.to(self.config.device)
        self.p_encoder = p_encoder.to(self.config.device)
        self.dense_embedding_matrix = None
        self.criterion = getattr(module_loss, self.config.loss)
        self.validation_step_outputs = {'sim_score': [], 'targets': []}
        self.metric_list = {metric: {"method": load(metric), "format": getattr(module_metric, metric)} for metric in self.config.metric.retriever}

    def configure_optimizers(self, ):
        trainable_params1 = list(filter(lambda p: p.requires_grad, self.q_encoder.parameters()))
        trainable_params2 = list(filter(lambda p: p.requires_grad, self.p_encoder.parameters()))
        trainable_params = [{"params": trainable_params1}, {"params": trainable_params2}]

        optimizer_name = self.config.optimizer.name
        del self.config.optimizer.name
        optimizer = init_obj(optimizer_name, self.config.optimizer, 
                             torch.optim, trainable_params)
        return optimizer

    def training_step(self, batch):
        targets = torch.zeros(self.config.data.batch_size).long().to(self.config.device)

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

        p_outputs = self.p_encoder(**p_inputs)
        q_outputs = self.q_encoder(**q_inputs)

        p_outputs = p_outputs.view(self.config.data.batch_size, self.config.data.num_neg+1, -1)
        q_outputs = q_outputs.view(self.config.data.batch_size, 1, -1)

        similarity_scores = torch.bmm(q_outputs, p_outputs.transpose(-2, -1)).squeeze()
        similarity_scores= F.log_softmax(similarity_scores, dim=-1)

        loss = self.criterion(similarity_scores, targets)
        return {"loss": loss}
            
    def validation_step(self, batch):
        targets = torch.zeros(self.config.data.batch_size).long().to(self.config.device)

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

        p_outputs = self.p_encoder(**p_inputs)
        q_outputs = self.q_encoder(**q_inputs)

        p_outputs = p_outputs.view(self.config.data.batch_size, self.config.data.num_neg+1, -1)
        q_outputs = q_outputs.view(self.config.data.batch_size, 1, -1)

        similarity_scores = torch.bmm(q_outputs, p_outputs.transpose(-2, -1)).squeeze()
        similarity_scores = F.log_softmax(similarity_scores, dim=-1)

        self.validation_step_outputs['sim_score'].extend(similarity_scores.cpu())
        self.validation_step_outputs['targets'].extend(targets.cpu())

    def on_validation_epoch_end(self, ):
        for k, v in self.validation_step_outputs.items():
            self.validation_step_outputs[k] = np.array(v).squeeze()

        # compute metric
        for name, metric in self.metric_list.items():
            output_format = metric['format'](self.validation_step_outputs)
            metric_result = metric['method'].compute(**output_format)
            print(name, metric_result)
        self.validation_step_outputs = {'sim_score': [], 'targets': []}

    def create_embedding_vector(self, ):
        self.p_encoder.eval()
        dataset_list = get_dataset_list(self.config.data.dataset_name)

        eval_dataset = concatenate_datasets([ds["validation"] for ds in dataset_list]).select(range(100))
        self.eval_corpus = list(set([example["context"] for example in eval_dataset]))

        p_embs = []
        for p in self.eval_corpus:
            p = self.tokenizer(p, padding='max_length', truncation=True, return_tensors='pt', max_length=self.config.data.max_seq_length)
            p_emb = self.p_encoder(**p)
            p_embs.append(p_emb.squeeze())
        
        self.dense_embedding_matrix = torch.stack(p_embs).T

    def search(self, query, k=1):
        self.q_encoder.eval()

        query_token = self.tokenizer([query], padding='max_length', truncation=True, return_tensors='pt', max_length=self.config.data.max_seq_length)
        query_vector = self.q_encoder(**query_token)

        similarity_score = torch.matmul(query_vector, self.dense_embedding_matrix).squeeze()
        sorted_result = torch.argsort(similarity_score, dim=-1, descending=True).squeeze()
        doc_scores = similarity_score[sorted_result]

        return doc_scores[:k], sorted_result[:k], self.eval_corpus[sorted_result[:k]]

