# Import necessary modules from LangChain and transformers
import argparse
import os
import torch
import gc
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain import PromptTemplate
import string

def arg_parser():
    parser = argparse.ArgumentParser(description="ThreeRiversRAG")

    parser.add_argument('--retrieval_dir', type=str, 
                        default="../data/rag_dummy_data/retrieval_data")
    parser.add_argument('--hf_cache_dir', type=str, default=None)
    parser.add_argument('--questions_file', type=str, 
                        default="../data/rag_dummy_data/questions.txt")
    parser.add_argument('--answers_file', type=str, 
                        default="../data/rag_dummy_data/answers.txt")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--generation_model_name", type=str,
                        default="google/flan-t5-base")
    parser.add_argument("--embedder_model_name", type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--annotation_data_format", type=str,
                        default="csv")
    parser.add_argument("--annotation_csv", type=str,
                        default="../data/annotation_data/annotation_qa/Culture.csv")
    parser.add_argument("--output_path", type=str,
                        default="../data/output/output.csv")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--db_path", type=str, default=None)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--no_prompt", action="store_true", default=False)
    parser.add_argument("--fewshot", action="store_true", default=False)
    parser.add_argument("--vectorstore_type", type=str, default="chroma", choices=["chroma", "faiss"],
                        help="Which vector store to use: chroma (default) or faiss.")
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
        pred = clean_answer(pred)
        gt = clean_answer(gt)
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

def extract_answer(answer):
    import re

    # Use a regex pattern to capture everything after "Helpful Answer:" until the end of the text
    match = re.search(r"(?i)Helpful Answer:\s*(.*)", answer, re.DOTALL)
    if match:
        answer_part = match.group(1).strip()
        answer_part = answer_part.split("\n")[0]
        return answer_part
    else:
        return "Extract failed"

def clean_answer(text):
    # Characters to strip from beginning and end
    edge_chars = set(string.whitespace + string.punctuation + '"\'')

    # Articles to optionally remove from the beginning
    leading_articles = {'the', 'a', 'an'}

    # Normalize input
    text = str(text).lower()
    text = text.replace(u'\u00a0', ' ')

    # Strip leading unwanted characters
    while text and text[0] in edge_chars:
        text = text[1:]
    # Strip trailing unwanted characters
    while text and text[-1] in edge_chars:
        text = text[:-1]

    # Remove all punctuation (including commas) from the text
    text = ''.join(char for char in text if char not in {",", "."})

    # Tokenize and remove leading article
    tokens = text.split()
    if tokens and tokens[0] in leading_articles:
        tokens = tokens[1:]

    return ' '.join(tokens)

def batch_inference(qa_chain, questions, batch_size, args, ground_truths=None):
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
        with torch.no_grad():
            if batch_size > 1:
                batch_outputs = qa_chain.batch(batch_inputs)
            else:
                batch_outputs = [qa_chain.invoke(batch[0])]
        torch.cuda.empty_cache()
        gc.collect()
        
        batch_predictions = []
        for output in batch_outputs:
            # Handle different output formats (dict or string)
            if isinstance(output, dict):
                if args.baseline:
                    answer = output.get("text", output)
                else:
                    answer = output.get("result", output)
            else:
                answer = output
            if args.generation_model_name in {"Qwen/Qwen2-7B-Instruct"}:
                answer = extract_answer(answer)
            batch_predictions.append(answer)
        
        predictions.extend(batch_predictions)
        
        # If ground truths are provided, compute and print batch metrics
        if ground_truths:
            batch_ground_truths = ground_truths[i:i+batch_size]
            batch_metrics = calculate_metrics(batch_predictions, batch_ground_truths)
            print(f"Batch {i//batch_size + 1}/{num_batches} Metrics:")
            for key, value in batch_metrics.items():
                print(f"  {key}: {value}")
        
        if batch_size == 1:
            print(f"Question: {batch[0]}")
            print(f"Answer: {batch_predictions[0]}")
            print(f"Ground Truth: {batch_ground_truths[0]}")
            print("-" * 40)
            
    return predictions

def main():
    args = arg_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")
    
    # Set HuggingFace cache directory if provided
    if args.hf_cache_dir:
        print(f"Setting HF cache directory to {args.hf_cache_dir}")
        os.environ['HF_HOME'] = args.hf_cache_dir

    if not args.db_path:
        # Load retrieval documents from the specified directory
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(args.retrieval_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, show_progress=True, use_multithreading=True)
        docs = loader.load()
        chunk_size = 1000
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=150)
        splitted_docs = splitter.split_documents(docs)
        print(f"Retrieval documents loaded.")

    # Initialize embeddings using a pre-trained model
    embeddings = HuggingFaceEmbeddings(model_name=args.embedder_model_name, model_kwargs={"device": device})

    # Build a vector store from the documents and embeddings.
    if args.vectorstore_type == "faiss":
        if not args.db_path:
            print("Building FAISS vector store...")
            vectorstore = FAISS.from_documents(splitted_docs, embeddings)
            vectorstore.save_local("faiss_index")
        else:
            vectorstore = FAISS.load_local(
                args.db_path, embeddings, allow_dangerous_deserialization=True
            )
    else:
        if not args.db_path:
            print(f"Building Chroma vector store...")
            emb = args.embedder_model_name.split("/")[-1]
            vectorstore = Chroma.from_documents(splitted_docs, embeddings, persist_directory=f"./chroma_db_chunk{chunk_size}_emb{emb}")
        else:
            print(f"Loading Chroma vectorstore")
            vectorstore = Chroma(persist_directory=args.db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.topk})
    print(f"Vector Store finish.")

    # Initialize a Hugging Face pipeline for text generation using an open-source model.
    tokenizer = AutoTokenizer.from_pretrained(args.generation_model_name)
    if args.generation_model_name == "Qwen/Qwen2-7B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(args.generation_model_name, torch_dtype=torch.float16, device_map="auto")
        model.eval()
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.float16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.generation_model_name, device_map="auto")
        # model.config.use_cache = False
        model.eval()
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create the RAG system using RetrievalQA
    # Define your custom instruction prompt
    ans_req = """Answer the question based only on the provided context in just one sentence. Each answer must be as concise as possible, extremely succinct—limited to just several keywords—and should not repeat the question."""
    yes_or_no_req = """If a question is a yes or no question, the answer must be exactly 'yes' or 'no' without any additional information and without punctuation"""
    if args.no_prompt:
        instruction_prompt = ""
    else:
        if args.generation_model_name == 'google/flan-t5-base':
            instruction_prompt = "Answer the question based only on the provided context in just one sentence."
        else:
            instruction_prompt = " ".join([ans_req, yes_or_no_req])

    
    if args.baseline:
        # update the prompt for baseline, no context
        ans_req = """Answer the question in just one sentence. Each answer must be as concise as possible, extremely succinct—limited to just several keywords—and should not repeat the question."""
        instruction_prompt = " ".join([ans_req, yes_or_no_req])
        template = (
            f"{instruction_prompt}\n"
            "Question: {question}\n"
            "Helpful Answer:"
        )
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=template
        )
        qa_chain = LLMChain(prompt=prompt_template, llm=llm)
    else:
        if args.fewshot:
            # Define a string with few-shot examples
            few_shot_examples = (
                "Example 1:\n"
                "Context: Carnegie Mellon University was founded in Pittsburgh in 1900.\n"
                "Question: When was Carnegie Mellon University founded?\n"
                "Answer: 1900\n\n"
                "Example 2:\n"
                "Context: UPMC Shadyside is part of the University of Pittsburgh system.\n"
                "Question: Is UPMC Shadyside part of the University of Pittsburgh system?\n"
                "Answer: Yes"
            )
            template = (
                f"{instruction_prompt}\n"
                "Few-shot examples:\n"
                f"{few_shot_examples}\n"
                "Context: {context}\n"
                "Question: {question}\n"
                "Helpful Answer:"
            )
        else:
            template = (
                f"{instruction_prompt}\n"
                "Context: {context}\n"
                "Question: {question}\n"
                "Helpful Answer:"
            )

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt_template})

    if args.annotation_data_format == "csv":
        # load questions and answers
        df = pd.read_csv(args.annotation_csv)
        if len(df) > 2500:
            df = df.sample(n=2500, replace=False, random_state=42)
        questions = df['Questions'].tolist()
        ground_truths = df['Answers'].tolist()

    elif args.annotation_data_format == "txt":
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
    predictions = batch_inference(qa_chain, questions, args.batch_size, args, ground_truths if ground_truths else None)
    print("Batch inference complete.\n")

    # Add the predictions as a new column in the DataFrame
    df['Predictions'] = predictions

    # Save the updated DataFrame to a new CSV file
    df.to_csv(args.output_path, index=False)

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

# todo: semantic text splitter, temperature