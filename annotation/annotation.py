import torch
from sympy.physics.units import temperature
import pandas as pd
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm

def retreive_documents(file):
    # Load the source link csv file
    file_links = pd.read_csv(file)
    dict_links = {}
    for index, row in file_links.iterrows():
        text_path = row['Source Data'].strip().replace(" ", "_") + ".txt"
        text_path = "$HOME/ThreeRiversRAG/data/annotation_data/main_source_link_data/" + text_path
        dict_links[(row['Source Data'], row["Topic"])] = text_path
    return dict_links


def retreive_questions(text):
    text = text.strip()
    find_qa = re.compile(r"Q: (.*?) Ans: (.*?)\n")
    questions = find_qa.findall(text)
    # print(type(questions))
    return questions

def main():

    # Load the model and tokenizer to the GPU
    gpu_properties = torch.cuda.get_device_properties(0)
    total_memory_gb = gpu_properties.total_memory / (1024 ** 3)
    print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    if torch.cuda.is_available():
        print("Using GPU")
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    # Define the pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.float16)

    intro_info = """You are a smart assistant designed to help come up with reading comprehension questions. You will be given a web-crawled document relevant to topics about Pittsburgh and Carnegie Mellon University (CMU) such as general information/history, events, music, sports, and culture."""
    task = """Based on the document, generate exactly 10 question and answer pairs covering different content topics."""
    requirement = """Each question must be independently answerable and the answers must be directly and exactly found in the document."""
    answer_format = """For each pair, output in this exact format without any extra text:
        Q: "YOUR_QUESTION_HERE", Ans: "YOUR_ANSWER_HERE"."""
    ans_req = """Do not include any introductory text, commentary, or explanations. The final output must contain only 10 Q/A pairs, nothing else. Each answer must be extremely succinct (only key words or phrases) and should not repeat the question."""
    examples = """Examples:
        Q: When was Carnegie Mellon University founded, Ans: 1900"""
    additional_info = """Example question and answer pairs are just sample, you need to generate your own questions based on the document content."""

    instruct_prompts = [intro_info, task, requirement, ans_req, answer_format, examples, additional_info]
    INSTRUCTIONS = " ".join(instruct_prompts)

    # Load the source links
    soruce_links = "$HOME/ThreeRiversRAG/data/annotation_data/main_source_link_data/source_data_links.csv"
    dict_links = retreive_documents(soruce_links)

    topics_df = {}
    for key, value in tqdm(dict_links.items(), desc="Processing items"):
        if key[1] not in topics_df:
            topics_df[key[1]] = pd.DataFrame(columns=["Questions", "Answers"])
        with open(value, "r") as file:
            document = file.read()
        input_prompts = INSTRUCTIONS + "\n\n" + "Document content: " + document

        messages = [
            {"role": "user", "content": input_prompts},
        ]
        with torch.no_grad():
            print("Generating questions...")
            if torch.cuda.is_available():
                print("Using GPU for inference...")
                model.to("cuda")
            try:
                result = pipe(messages, max_new_tokens=512, temperature = 0.8, top_k = 50, top_p = 0.95)
            except:
                print("Generation failed", key[0])
                continue

        try:
            questions = retreive_questions(result[0]['generated_text'][1]["content"])
        except:
            print("Failed to retrieve questions", key[0])
            continue
        for q, a in questions:
            topics_df[key[1]] = topics_df[key[1]].append({"Questions": q, "Answers": a}, ignore_index=True)

    # Save the questions to a csv file
    try:
        for key, value in topics_df.items():
            value.to_csv(f"~/ThreeRiversRAG/data/annotation_data/annotation_qa/{key[1]}.csv", index=False)
    except:
        print("Failed to save the questions to a csv file")


if __name__ == "__main__":
    main()
