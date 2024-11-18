# Containerization and Orchestration of Information Retrieval Experiments

## Overview
The experiments evaluate the performance of a retrieval system based on Sentence Transformers and FAISS. The system is run using Docker Compose to ensure easy and consistent execution of the experiment environment.


### Steps to Run the Experiment

1. Clone this repository:
    ```bash
    git clone https://github.com/paoloitaliani/contenerized-retrieval.git
    cd contenerized-retrieval
    ```

2. Build and run the Docker container using Docker Compose:
    ```bash
    docker-compose up --build
    ```

3. The experiment will automatically start and the results will be saved in the `./outputs` directory.


## Experiment Description

The **SciFact** dataset is used. The retrieval system is evaluated based on how well it retrieves the correct document from a corpus given a query.

- **Model**: The default model used for generating embeddings is `all-mpnet-base-v2`. You can change the model by setting the `MODEL` environment variable.
- **Embeddings**: The embeddings are generated using the SentenceTransformer library and are normalized with FAISS for efficient similarity search.
- **Indexing**: FAISS is used for indexing the embeddings, and cosine similarity is used for retrieval.
- **Metrics**: The Rank@N metric is used to evaluate the results, that is the percentage of instances where the correct document is retrieved within the top N results.


