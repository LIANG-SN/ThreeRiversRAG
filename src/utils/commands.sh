python rag.py --retrieval_dir /home/ubuntu/ThreeRiversRAG/data/data_all/data/sample_data_crawled_static_web_data

python rag.py --retrieval_dir /home/ubuntu/ThreeRiversRAG/data/retrieve_source  --annotation_csv /home/ubuntu/ThreeRiversRAG/data/annotation_data/annotation_qa/qa2500.csv --embedder_model_name 'sentence-transformers/all-MiniLM-L6-v2'

python rag.py --retrieval_dir /home/ubuntu/ThreeRiversRAG/data/retrieve_source  --annotation_csv /home/ubuntu/ThreeRiversRAG/data/annotation_data/annotation_qa/annotation_test_v1.csv  --embedder_model_name 'sentence-transformers/all-MiniLM-L6-v2' --generation_model_name 'Qwen/Qwen2-7B-Instruct'

python rag.py --retrieval_dir /home/ubuntu/ThreeRiversRAG/data/retrieve_source  --annotation_csv /home/ubuntu/ThreeRiversRAG/data/annotation_data/annotation_qa/annotation_selected_100.csv  --embedder_model_name 'sentence-transformers/all-MiniLM-L6-v2' --generation_model_name 'Qwen/Qwen2-7B-Instruct' --db_path ./chroma_db_chunk500_mini

python rag.py --retrieval_dir /home/ubuntu/ThreeRiversRAG/data/retrieve_source  --annotation_csv /home/ubuntu/ThreeRiversRAG/data/annotation_data/annotation_qa/annotation_selected_100.csv  --embedder_model_name 'sentence-transformers/all-MiniLM-L6-v2' --generation_model_name 'Qwen/Qwen2-7B-Instruct' --vectorstore_type faiss --db_path faiss_index --fewshot