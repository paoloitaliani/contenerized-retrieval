import os
import torch
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL = os.getenv('MODEL', "all-mpnet-base-v2")


def rank_k(indexes, k, target_idx):
    if target_idx in indexes[:k]:
        return 1
    else:
        return 0


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = load_dataset("mteb/scifact", "corpus")["corpus"]
    dataset_text = [x["text"] for x in dataset]
    dataset_query = [x["title"] for x in dataset]
    dataset_ids = [i for i, _ in enumerate(dataset)]
    

    model = SentenceTransformer(MODEL).to(device)
    embeddings = model.encode(dataset_text, show_progress_bar=True, batch_size=2)
    
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIDMap(index)

    index.add_with_ids(embeddings, np.array(dataset_ids))

    rank_1_list = []
    rank_5_list = []
    rank_10_list = []
    for i, query_text in tqdm(enumerate(dataset_query)):
        query = model.encode([query_text]) 
        # compute cosine similarities scores
        faiss.normalize_L2(query)
        cos_similarities, faiss_idx = index.search(np.array(query).astype("float32"), k=10)
        retrieved_indexes = faiss_idx[0]
        rank_1_list.append(rank_k(retrieved_indexes, 1, i))
        rank_5_list.append(rank_k(retrieved_indexes, 5, i))
        rank_10_list.append(rank_k(retrieved_indexes, 10, i))
    
    print("Results:")
    print(f"Rank 1: {round(100 * sum(rank_1_list)/len(dataset_query), 2)}")
    print(f"Rank 5: {round(100 * sum(rank_5_list)/len(dataset_query), 2)}")
    print(f"Rank 10: {round(100 * sum(rank_10_list)/len(dataset_query), 2)}")

if __name__ == "__main__":
    main()

