import pandas as pd
import argparse
from sklearn.metrics import cohen_kappa_score
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def arg_parser():
    parser = argparse.ArgumentParser(description="ThreeRiversRAG")

    parser.add_argument('--gold_test', type=str,
                        default="./annotation_test_clean.csv",)
    parser.add_argument('--generate_sample', type=str,
                        default="false")
    parser.add_argument("--annotation_result", type=str,
                        default="./annotation_result.csv")
    return parser.parse_args()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    normalized_text = " ".join(tokens)
    return normalized_text

def preprocess_quality(value):
    value = int(value)
    if value >= 3:
        return 1
    else:
        return 0


def cosine_similarity_pair(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Random select 50 questions from the annotation data
def main():
    args = arg_parser()
    generate_sample = args.generate_sample
    if generate_sample == "true":
        annotation_data = pd.read_csv(args.gold_test)
        # only select the data that Source index is positive
        annotation_data = annotation_data[annotation_data["Source_Index"] > 0]
        random_sample = annotation_data.sample(n=50)

        # Save the random sample to a csv file
        random_sample.to_csv("random_sample.csv", index=False)
    else:
        print("No random sample generated")

    # Load the annotation result
    annotation_result = pd.read_csv(args.annotation_result)
    # convert all the answers to string in lowercase and strip the white spaces
    annotation_result['Annotation 1'] = annotation_result['Annotation 1'].apply(preprocess_text)
    annotation_result['Annotation 2'] = annotation_result['Annotation 2'].apply(preprocess_text)
    kappa = cohen_kappa_score(annotation_result['Annotation 1'], annotation_result['Annotation 2'])
    print("\nCohen's Kappa Score for the QA answering correctness:", kappa)

    annotation_result['cosine_similarity'] = annotation_result.apply(
        lambda row: cosine_similarity_pair(row['Annotation 1'], row['Annotation 2']), axis=1
    )

    average_similarity = annotation_result['cosine_similarity'].mean()
    print("\nAverage Cosine Similarity:", average_similarity)

    # convert all the answers to string in lowercase and strip the white spaces
    annotation_result['Label 1'] = annotation_result['Quality 1'].apply(preprocess_quality)
    annotation_result['Label 2'] = annotation_result['Quality 2'].apply(preprocess_quality)
    kappa = cohen_kappa_score(annotation_result['Quality 1'], annotation_result['Quality 2'])
    print("\nCohen's Kappa Score for the QA quality:", kappa)
    kappa = cohen_kappa_score(annotation_result['Label 1'], annotation_result['Label 2'])
    print("\nCohen's Kappa Score for the QA quality category:", kappa)



if __name__ == "__main__":
    main()