import pandas as pd
from tqdm import tqdm
import argparse
import json
def arg_parser():
    parser = argparse.ArgumentParser(description="ThreeRiversRAG")

    parser.add_argument('--annotation_dir', type=str,
                        default="./annotation_test_clean.csv",)
    return parser.parse_args()

def main():

    args = arg_parser()
    annotations = pd.read_csv(args.annotation_dir)
# Generate the reference questions
    with open("questions.txt", "w") as file:
        for i in tqdm(range(annotations.shape[0])):
            file.write(f"{i + 1}: " + annotations.iloc[i]["Questions"])
            file.write("\n")

# Generate the reference answers
    answers = {}
    for i in tqdm(range(annotations.shape[0])):
        answers[str(i + 1)] = annotations.iloc[i]["Answers"]
    # Save the answers to a json file

    with open("answers.json", "w") as json_file:
        json.dump(answers, json_file, indent=4)


if __name__ == "__main__":
    main()
