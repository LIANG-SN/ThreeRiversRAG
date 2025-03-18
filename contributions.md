# Contributions
- **Data creation:**
Data creation involved web scraping to collect relevant textual data for the project. Initially, all three team members collaborated to identify and gather raw source web links. Each team member was responsible for different aspects of data collection, utilizing BeautifulSoup for static web scraping and Selenium for dynamic web scraping.
  - Muyang Xu: Implemented static web scraping using BeautifulSoup to collect data related to the General Info and History of CMU.
  - Shengnan Liang: Implemented static web scraping using BeautifulSoup to collect data related to the General Info and History of Pittsburgh.
  - Xinru Li: Implemented both static web scraping with BeautifulSoup and dynamic web scraping with Selenium to collect data on Events in Pittsburgh and CMU, Music and Culture, and Sports.


- **Data Annotation:** 
  - Muyang Xu: 
    - Developed a comprehensive annotation guideline and prompt engineering strategies for LLM-based auto annotation.
    - Generated a gold standard test dataset for the general info and history of Pittsburgh and CMU, music and culture.
    - Conducted an IAA evaluation to ensure the quality of the annotated data.
  - Shengnan Liang: 
    - Developed a comprehensive annotation guideline and prompt engineering strategies for LLM-based auto annotation.
    - Generated a gold standard test dataset for events and sport related to CMU and Pittsburgh.
  - Xinru Li:
    - Developed a comprehensive annotation guideline and prompt engineering strategies for LLM-based auto annotation.
    - Generated a gold standard test dataset for food related to CMU and Pittsburgh.
    - Conducted an IAA evaluation to ensure the quality of the annotated data.