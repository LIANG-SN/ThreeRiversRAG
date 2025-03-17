# ThreeRiversRAG

Retrieval-Augmented Generation (RAG) is a powerful solution to the knowledge limitations of large language models (LLMs), especially for domain-specific question answering. In this project, we design and implement a full RAG pipeline focused on answering questions about Pittsburgh and Carnegie Mellon University (CMU). Our system includes comprehensive data collection, preprocessing, and annotation, with quality evaluated using Inter-Annotator Agreement (IAA). We compare large-scale models (e.g., Llama, Qwen) with smaller ones to assess their effectiveness in knowledge-intensive tasks. Additionally, we analyze how retrieval strategies and prompt design influence the accuracy and relevance of generated answers, underscoring the importance of retrieval and prompt tuning in domain-specific QA systems.

## RAG Pipeline

<p align="center">
  <img src="./src/plots/rag_pipeline.png" alt="RAG Pipeline Diagram" width="300"/>
</p>

### Steps to run

1. Download the retrieval source data and place it in data folder.
2. Enter src folder: `cd src`.
3. Install the relevant packages: `pip install -r requirements.txt`
4. Run RAG pipeline
    ```
    python rag.py \
    --retrieval_dir ../data/retrieve_source \
    --annotation_csv ../data/test/annotation_test_v3.csv \
    --embedder_model_name 'sentence-transformers/all-MiniLM-L6-v2' \
    --generation_model_name 'Qwen/Qwen2-7B-Instruct' \
    --vectorstore_type faiss
    ```