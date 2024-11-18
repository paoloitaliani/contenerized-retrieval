import os
import sys
import torch
import faiss
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
from hashlib import sha512

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


MODEL = os.getenv('MODEL', "all-mpnet-base-v2")
# Compute a unique experiment ID based on the hyper-parameters
FOOTPRINT_KEYS = {"MODEL"}
EXPERIMENT_FOOTPRINT = {k: v for k, v in locals().items() if k in FOOTPRINT_KEYS}
EXPERIMENT_FOOTPRINT_YAML = yaml.dump(EXPERIMENT_FOOTPRINT, sort_keys=True)
EXPERIMENT_ID = sha512(EXPERIMENT_FOOTPRINT_YAML.encode()).hexdigest()

# Ensure the output directory exists
DATA_DIR = Path(os.getenv('DATA_DIR', './output_folder'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(DATA_DIR)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rank_k(indexes, k, target_idx):
    if target_idx in indexes[:k]:
        return 1
    else:
        return 0


def main():
    # Do not waste time if the experiment has already been run
    # by looking for the existence of a .yaml file named after the EXPERIMENT_ID
    for file in DATA_DIR.glob('**/*.yaml'):
        if file.is_file() and file.stem == EXPERIMENT_ID:
            print(f'Experiment with ID {EXPERIMENT_ID[:8]} already exists in {file.parent}', file=sys.stderr)
            exit(0)
    
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

    # Save experiment footprint into OUTPUT_DIR/EXPERIMENT_ID.yaml
    with open(OUTPUT_DIR / f'{EXPERIMENT_ID}.yaml', 'w') as file:
        yaml.dump(EXPERIMENT_FOOTPRINT, file)

if __name__ == "__main__":
    main()

