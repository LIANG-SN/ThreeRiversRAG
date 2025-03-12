# Import necessary modules from LangChain and transformers
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from transformers import pipeline
import argparse
from langchain_community.document_loaders import DirectoryLoader


def arg_parser():
    parser = argparse.ArgumentParser(description="ThreeRiversRAG")

    parser.add_argument('--retrieval_dir', type=str, default="../data/retrieval_data")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = arg_parser()
    
    loader = DirectoryLoader(args.retrieval_dir)
    docs = loader.load()

    # Initialize HuggingFace embeddings using a pre-trained model from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Build a Chroma vector store from the documents and embeddings.
    # The persist_directory parameter is optional; it saves the index on disk.
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()

    # Initialize a Hugging Face pipeline for text generation using an open-source model.
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create a Retrieval-Augmented Generation (RAG) system by combining the retriever and the LLM.
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Ask a question and print the answer
    question = "What is Pittsburgh known for?"
    answer = qa_chain.run(question)
    print("Question:", question)
    print("Answer:", answer)
