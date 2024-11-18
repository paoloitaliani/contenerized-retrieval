FROM llm_image:latest
RUN mkdir -p /contenerized-retrieval
VOLUME "/outputs"
ENV DATA_DIR=/outputs
WORKDIR /contenerized-retrieval
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY * /contenerized-retrieval/
ENV OWNER=1124:1124
CMD export OUTPUT_DIR=$DATA_DIR/$(date +%Y-%m-%d-%H-%M-%S)-$(hostname) && \
    mkdir -p $OUTPUT_DIR && \
    python3 perform_retrieval.py | tee $OUTPUT_DIR/output.log && \
    chown -R $OWNER $DATA_DIR