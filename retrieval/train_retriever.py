import json
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

from DenseRetrieval import DenseRetrieval, BertEncoder

def prepare_in_batch_negative(dataset=None, num_neg=2, tokenizer=None, batch_size=2):
        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        passage_dataloader = DataLoader(passage_dataset, batch_size=batch_size)

        return train_dataloader, passage_dataloader

def main():
    model_checkpoint = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # 데이터셋과 모델은 아래와 같이 불러옵니다.
    train_dataset = load_dataset("squad_kor_v1")['train']
    
    # 메모리가 부족한 경우 일부만 사용하세요 !
    batch_size = 8
    num_sample = 20000
    sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
    train_dataset = train_dataset[sample_idx]
    num_neg = 2
    train_dataloader, passage_dataloader = prepare_in_batch_negative(dataset = train_dataset, num_neg=num_neg, tokenizer=tokenizer, batch_size=batch_size)

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01
    )

    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    retriever = DenseRetrieval(tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in retriever.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in retriever.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in retriever.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in retriever.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0

    retriever.p_encoder.zero_grad()
    retriever.q_encoder.zero_grad()
    torch.cuda.empty_cache()

    for _ in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                retriever.p_encoder.train()
                retriever.q_encoder.train()

                targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                targets = targets.to(args.device)

                p_inputs = {
                    'input_ids': batch[0].view(batch_size * (num_neg + 1), -1).to(args.device),
                    'attention_mask': batch[1].view(batch_size * (num_neg + 1), -1).to(args.device),
                    'token_type_ids': batch[2].view(batch_size * (num_neg + 1), -1).to(args.device)
                }

                q_inputs = {
                    'input_ids': batch[3].to(args.device),
                    'attention_mask': batch[4].to(args.device),
                    'token_type_ids': batch[5].to(args.device)
                }

                p_outputs = retriever.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                q_outputs = retriever.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                # Calculate similarity score & loss
                p_outputs = p_outputs.view(batch_size, num_neg + 1, -1)
                q_outputs = q_outputs.view(batch_size, 1, -1)

                sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                sim_scores = sim_scores.view(batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)
                tepoch.set_postfix(loss=f'{str(loss.item())}')

                loss.backward()
                optimizer.step()
                scheduler.step()

                retriever.p_encoder.zero_grad()
                retriever.q_encoder.zero_grad()

                global_step += 1
                torch.cuda.empty_cache()
                del p_inputs, q_inputs

    if not os.path.exists("./p_encoder"):
        os.makedirs("./p_encoder")
    if not os.path.exists("./q_encoder"):
        os.makedirs("./q_encoder")
    retriever.p_encoder.save_pretrained("./p_encoder")
    retriever.q_encoder.save_pretrained("./q_encoder")


if __name__ == "__main__":
    main()
