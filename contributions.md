# Contributions

## Muyang Xu(muyangxu)

- **Implemented the Automated Data Annotation:**
  - Developed prompt engineering strategies for the LLM.
  - Designed and implemented the retrieval and formatting of QA pairs.
  - Generated a gold standard test dataset for the Retrieval-Augmented Generation (RAG) pipeline.
  - Initiated the IAA evaluation by:
    - Implementing Cohen's Kappa for measuring categorical agreement.
    - Developing a semantic consistency similarity metric using cosine similarity with TF-IDF vectorization.
   

## Data Creation
Data creation involved web scraping to collect relevant textual data for the project. Initially, all three team members collaborated to identify and gather raw source web links. Each team member was responsible for different aspects of data collection, utilizing BeautifulSoup for static web scraping and Selenium for dynamic web scraping.
  - Muyang Xu: Implemented static web scraping using BeautifulSoup to collect data related to the General Info and History of CMU.
  - Shengnan Liang: Implemented static web scraping using BeautifulSoup to collect data related to the General Info and History of   Pittsburgh.
  - Xinru Li: Implemented both static web scraping with BeautifulSoup and dynamic web scraping with Selenium to collect data on Events in Pittsburgh and CMU, Music and Culture, and Sports.

## Modeling

- Muyang Xu: Researched and implemented retrievers such as FAISS and Chroma, and experiment with few shot learning.
- Xinru Li: Researched and implemented embedding models such as `all-mpnet-base-v2` and `sall-MiniLM-L6-v2` and perform output analysis.
- Shengnan Liang: Researched and implemented generation models such as Qwen and T5, developed the `rag.py` and relevant util scripts.