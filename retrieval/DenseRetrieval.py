import time
import os
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Union
from contextlib import contextmanager

import pickle
import numpy as np
import pandas as pd
from datasets import Dataset

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from transformers import (
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        return pooled_output
    

class DenseRetrieval:
    def __init__(
        self, p_encoder, q_encoder, tokenizer, 
        data_path: Optional[str] = "../../data/",
        context_path: Optional[str] = "wikipedia_documents.json"
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder.to(self.device)
        self.q_encoder = q_encoder.to(self.device)
        self.data_path = data_path
        
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        self.contexts = self.contexts
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        

    def get_dense_embedding(self):
        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"dense_embedding.pt"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.build_embedding(emd_path)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def build_embedding(self, save_path):
        print('tokenizing....')
        tokenized_contexts = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        print('tokenizing done')
        contexts_dataset = TensorDataset(
            tokenized_contexts['input_ids'], tokenized_contexts['attention_mask'], tokenized_contexts['token_type_ids']
        )
        self.contexts_dataloader = DataLoader(contexts_dataset, batch_size=256)
        
        with torch.no_grad():
            self.p_encoder.eval()
            p_embs = []
            for batch in tqdm(self.contexts_dataloader, desc='build dense embedding....'):
                batch = tuple(t.to(self.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = self.p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.cat(p_embs, dim=0).view(len(self.contexts_dataloader.dataset), -1)  # (num_passage, emb_dim)
        torch.save(p_embs, save_path)
        return p_embs

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        print('query tokenizing....')
        tokenized_queries = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt')
        print('query tokenizing done')
        q_dataset = TensorDataset(
            tokenized_queries['input_ids'], tokenized_queries['attention_mask'], tokenized_queries['token_type_ids']
        )
        self.q_dataloader = DataLoader(q_dataset, batch_size=256)
        
        with torch.no_grad():
            self.q_encoder.eval()
            q_embs = []
            for batch in tqdm(self.q_dataloader, desc='query dense embedding....'):
                batch = tuple(t.to(self.device) for t in batch)
                q_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                q_emb = self.q_encoder(**q_inputs).to('cpu')  # (num_query=1, emb_dim)
                q_embs.append(q_emb)

        query_vec = torch.cat(q_embs, dim=0).view(len(queries), -1)  # (num_query, emb_dim)
        
        result = torch.matmul(query_vec, torch.transpose(self.p_embedding, 0, 1))
        if not isinstance(result, np.ndarray):
            result = result.numpy()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices



    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):
        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(self.device)
            q_emb = self.q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embedding, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        doc_score = dot_prod_scores.squeeze()[rank].tolist()[:k]
        doc_indices = rank.tolist()[:k]
        return doc_score, doc_indices