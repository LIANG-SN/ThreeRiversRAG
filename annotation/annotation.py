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
        text_path = "/home/ubuntu/ThreeRiversRAG/data/annotation_data/main_source_link_data/" + text_path
        dict_links[(row['Source Data'], row["Topic"])] = text_path
    return dict_links


def retreive_questions(text):
    text = text.strip()
    find_qa = re.compile(r"Q:\s*(.*?)\s*Ans:\s*(.*?)(?:\n|$)")
    questions = find_qa.findall(text)
    # print(type(questions))
    return questions

def main():
    max_length = 131072
    # Load the model and tokenizer to the GPU
    gpu_properties = torch.cuda.get_device_properties(0)
    total_memory_gb = gpu_properties.total_memory / (1024 ** 3)
    print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    if torch.cuda.is_available():
        print("Using GPU")
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, truncation=True, max_length=max_length)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    # Define the pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.float16)

    intro_info = """You are a smart assistant designed to help come up with reading comprehension questions. You will be given a web-crawled document relevant to topics about Pittsburgh and Carnegie Mellon University (CMU) such as general information/history, events, music, sports, and culture."""
    task = """Based on the document, generate 10 question and answer pairs that cover diverse topics and aspects from the content."""
    requirement = """Each question must be independently answerable and the answers must be directly and exactly found in the document."""
    additional_req = """Questions must be specific and focused. They should cover a range of aspects such as specific events, people, dates, locations, and other relevant details."""
    question_restriction = """Avoid broad, vague, open-ended questions that would result in long, narrative answers. For example, instead of asking 'What can you tell me about the event?', ask a question that targets a particular detail of the event."""
    ans_req = """Do not include any introductory text, commentary, or explanations. The final output must contain only 10 Q/A pairs, nothing else. """
    additional_ans_req = """Each answer must be extremely succinct—limited to just several keywords—and should not repeat the question."""
    yes_or_no_req = """If a question is a yes or no question, the answer must be exactly 'yes' or 'no' without any additional information."""
    answer_format = """For each pair, output in this exact format without any extra text:
                Q: "YOUR_QUESTION_HERE", Ans: "YOUR_ANSWER_HERE"."""
    examples = """Examples: Q: When was Carnegie Mellon University founded, Ans: 1900\nQ:When does Kara Walker exhibition open?, Ans: March 1\nQ: "Is the event held indoors?", Ans: "yes"."""
    additional_info = """The example pairs are for illustration only; you must generate original Q/A pairs based solely on the document content."""


    instruct_prompts = [intro_info, task, requirement, additional_req, question_restriction, ans_req, additional_ans_req, yes_or_no_req, answer_format, examples, additional_info]
    INSTRUCTIONS = " ".join(instruct_prompts)

    # Load the source links
    soruce_links = "/home/ubuntu/ThreeRiversRAG/data/annotation_data/main_source_link_data/source_data_links.csv"
    dict_links = retreive_documents(soruce_links)

    topics_df = {}
    for key, value in tqdm(dict_links.items(), desc="Processing items"):
        SOURCE, TOPIC = key
        if TOPIC not in topics_df:
            topics_df[TOPIC] = pd.DataFrame(columns=["Questions", "Answers", "Source"])
        try:
            with open(value, "r") as file:
                document = file.read()
        except:
            print("Failed to read/crawl the document", SOURCE)
            continue
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
                result = pipe(messages, max_new_tokens=512, temperature = 1, top_k = 50, top_p = 0.95)
            except:
                print("Generation failed", SOURCE)
                continue

        try:
            questions = retreive_questions(result[0]['generated_text'][1]["content"])
        except:
            print("Failed to retrieve questions", SOURCE)
            continue

        # TODO: Figure out why some annotations cannot retrieve questions??? len(questions) == 0????
        print("Questions generated successfully:", SOURCE, len(questions))
        if len(questions) < 10:
            with open("failed_annotations.txt", "a") as file:
                file.write(f"{SOURCE}:\n" + result[0]['generated_text'][1]["content"] + "\n\n")
        for q, a in questions:
            new_row = pd.DataFrame({"Questions": [q], "Answers": [a], "Source": [SOURCE]})
            topics_df[TOPIC] = pd.concat([topics_df[TOPIC], new_row], ignore_index=True)
        print("QA stored successfully..........")
    # Save the questions to a csv file
    try:
        for key, value in tqdm(topics_df.items(), desc="Saving files"):
            value.to_csv(f"/home/ubuntu/ThreeRiversRAG/data/annotation_data/annotation_qa/{key}.csv", index=False)
    except:
        print("Failed to save the questions to a csv file")


if __name__ == "__main__":
    main()
