# Import necessary modules from LangChain and transformers
import argparse
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from transformers import pipeline
from langchain_community.document_loaders import DirectoryLoader


def arg_parser():
    parser = argparse.ArgumentParser(description="ThreeRiversRAG")

    parser.add_argument('--retrieval_dir', type=str, 
                        default="../data/rag_dummy_data/retrieval_data")
    parser.add_argument('--hf_cache_dir', type=str, default=None)
    parser.add_argument('--questions_file', type=str, 
                        default="../data/rag_dummy_data/questions.txt")
    parser.add_argument('--answers_file', type=str, 
                        default="../data/rag_dummy_data/answers.txt")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for inference")
    
    return parser.parse_args()

def calculate_metrics(predictions, ground_truths):
    """
    Calculates evaluation metrics (recall, F1, exact match) for a list of predictions
    compared to ground truths.
    """
    # Ensure the lengths match
    assert len(predictions) == len(ground_truths), "Lengths of predictions and ground truths must match."
    total_examples = len(predictions)
    
    exact_match_count = 0
    total_recall = 0.0
    total_precision = 0.0

    for pred, gt in zip(predictions, ground_truths):
        # Count exact matches
        if pred == gt:
            exact_match_count += 1
        
        # Tokenize the prediction and ground truth
        pred_tokens = set(pred.split())
        gt_tokens = set(gt.split())
        
        # Compute intersection of tokens
        common_tokens = pred_tokens & gt_tokens
        
        # Compute recall and precision for the current pair
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        
        total_recall += recall
        total_precision += precision

    # Compute average recall and precision
    avg_recall = total_recall / total_examples
    avg_precision = total_precision / total_examples
    
    # Calculate F1 score (harmonic mean)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    exact_match = exact_match_count / total_examples

    return {
        "recall": avg_recall,
        "f1": f1_score,
        "exact_match": exact_match,
        "exact_match_count": exact_match_count,
        "total_examples": total_examples,
        "total_recall": total_recall,
        "total_precision": total_precision
    }

def batch_inference(qa_chain, questions, batch_size, ground_truths=None):
    """
    Processes the list of questions in batches and returns accumulated predictions.
    If ground_truths are provided, computes and prints metrics for each batch.
    """
    predictions = []
    num_batches = (len(questions) + batch_size - 1) // batch_size

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_inputs = [{"query": q} for q in batch]
        # Use invoke to process the batch
        batch_outputs = qa_chain.batch(batch_inputs)
        
        batch_predictions = []
        for output in batch_outputs:
            # Handle different output formats (dict or string)
            if isinstance(output, dict):
                answer = output.get("result", output)
            else:
                answer = output
            batch_predictions.append(answer)
        
        predictions.extend(batch_predictions)
        
        # If ground truths are provided, compute and print batch metrics
        if ground_truths:
            batch_ground_truths = ground_truths[i:i+batch_size]
            batch_metrics = calculate_metrics(batch_predictions, batch_ground_truths)
            print(f"Batch {i//batch_size + 1}/{num_batches} Metrics:")
            for key, value in batch_metrics.items():
                print(f"  {key}: {value}")
            print("-" * 40)
            
    return predictions

def main():
    args = arg_parser()
    
    # Set HuggingFace cache directory if provided
    if args.hf_cache_dir:
        print(f"Setting HF cache directory to {args.hf_cache_dir}")
        os.environ['HF_HOME'] = args.hf_cache_dir

    # Load retrieval documents from the specified directory
    loader = DirectoryLoader(args.retrieval_dir)
    docs = loader.load()

    # Initialize embeddings using a pre-trained model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Build a Chroma vector store from the documents and embeddings.
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()

    # Initialize a Hugging Face pipeline for text generation using an open-source model.
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create the RAG system using RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Load questions from file
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(questions)} questions.")

    # Load ground truth answers if provided
    ground_truths = []
    if args.answers_file and os.path.exists(args.answers_file):
        with open(args.answers_file, "r", encoding="utf-8") as f:
            ground_truths = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(ground_truths)} answers.")

    # Perform batch inference with customizable batch size
    predictions = batch_inference(qa_chain, questions, args.batch_size, ground_truths if ground_truths else None)
    print("Batch inference complete.\n")

    # Print overall predictions
    for question, answer in zip(questions, predictions):
        print("Question:", question)
        print("Answer:", answer)
        print("-" * 40)

    # Compute and print overall metrics if ground truths are provided
    if ground_truths:
        if len(ground_truths) != len(predictions):
            print("Warning: The number of ground truths does not match the number of predictions.")
        else:
            overall_metrics = calculate_metrics(predictions, ground_truths)
            print("\nOverall Evaluation Metrics:")
            for key, value in overall_metrics.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()