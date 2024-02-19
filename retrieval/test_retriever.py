import random
import argparse
import time
from contextlib import contextmanager

from transformers import AutoTokenizer
import numpy as np
from datasets import concatenate_datasets, load_from_disk

from SparseRetrieval import SparseRetrieval

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="../data/train_dataset", type=str, help="",
        default="../data/train_dataset"
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
        default="bert-base-multilingual-cased"
    )
    parser.add_argument("--data_path", metavar="../data", type=str, help="", default="../data")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help="",
        default="wikipedia_documents.json"
    )
    parser.add_argument("--retrieve_type", default="sparse")
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="", default=False)

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)

    if args.retrieve_type == "dense":
        pass
        # retriever = DenseRetrieval(
        #     tokenizer=tokenizer,
        #     data_path=args.data_path,
        #     context_path=args.context_path,
        # )
    else:
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        retriever.get_sparse_embedding()

    if args.use_faiss:
        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

        # test single query
        # with timer("single query by faiss"):
        #     query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
        #     scores, indices = retriever.retrieve_faiss(query)

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print("correct retrieval result by exhaustive search", df["correct"].sum() / len(df))

        # with timer("single query by exhaustive search"):
        #     query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"    
        #     scores, indices = retriever.retrieve(query)
            
    print("Retrieval Performance", df["correct"].sum(), "/", len(df))