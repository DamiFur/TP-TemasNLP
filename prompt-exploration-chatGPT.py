import pandas as pd
from glob import glob
from openai import OpenAI
from sklearn.metrics import f1_score
import argparse

parser = argparse.ArgumentParser(description="Annotate Hate Speech Dataset using ChatGPT")
parser.add_argument("--fewshot_samples", type=int, default=10, help="Number of fewshot samples to use")
parser.add_argument("--test_sample_size", type=int, default=30, help="Number of samples to test the model with")
parser.add_argument("--prompt_testing", type=bool, default=False, help="Test the prompt generation")
args = parser.parse_args()

dataset = pd.read_csv("Datasets/D3 - Anotación Original.tsv", sep="\t")
client = OpenAI()
sample_all = dataset.sample(args.test_sample_size + args.fewshot_samples, random_state=43)
FEWSHOT_SAMPLES = sample_all.iloc[:args.fewshot_samples, :]
sample = sample_all.iloc[args.fewshot_samples:, :]

if not args.prompt_testing:
    d1 = pd.read_csv("Datasets/D1 - Anotación Original.tsv", sep="\t")
    d2 = pd.read_csv("Datasets/D2 - Anotación Original.tsv", sep="\t")
    to_label_d1 = {"dataset": d1[d1["nro"] > 120], "name": "D1"}
    to_label_d2 = {"dataset": d2[d2["nro"] > 120], "name": "D2"}

PROMPT = ""

categories = ["HATEFUL", "WOMEN", "LGBTI", "RACISM", "CLASS", "POLITICS", "DISABLED", "APPEARANCE", "CRIMINAL"]

def generate_response(row):
    response = "NOT HATEFUL"
    if row["HATEFUL"] == 1:
        response = "HATEFUL"
        if row["WOMEN"] == 1:
            response += ", WOMEN"
        if row["LGBTI"] == 1:
            response += ", LGBTI"
        if row["RACISM"] == 1:
            response += ", RACISM"
        if row["CLASS"] == 1:
            response += ", CLASS"
        if row["POLITICS"] == 1:
            response += ", POLITICS"
        if row["DISABLED"] == 1:
            response += ", DISABLED"
        if row["APPEARANCE"] == 1:
            response += ", APPEARANCE"
        if row["CRIMINAL"] == 1:
            response += ", CRIMINAL"
    return response

def get_prompt(row, few_shot=[]):
    # print(row)
    prompt = [
        {"role": "system", "content": "You are an expert in hate speech detection in Spanish language. You are given a dataset of news headlines and reader comments in Spanish. You are asked to classify the comments as 'HATEFUL' or 'NO HATEFUL' based on whether the comment is hateful towards a protected group or community or not. You also have to specify if the hate message is targeted against one of the following possible targets: 'WOMEN' (hate is directed to someone just for being a woman), 'LGBTI' (hate is directed to someone for being part of the LGBTI community), 'RACISM' (hate is directed to someone because of it's race), 'CLASS' (hate is directed to someone because it's social class or economic status), 'POLITICS' (hate is directed to someone because of personal politic beliefs or ideology), 'DISABLED' (hate is directed to someone because of physical disability or problems with addictions), 'APPEARENCE' (hate is directed to someone because of physical appearence), 'CRIMINAL' (hate is directed to someone because it has criminal records). Respond only with the word 'HATEFUL' if the comment is HATEFUL or with 'NOT HATEFUL' if the comment is NOT HATEFUL. Add the targets of the hate message separated by commas if any. Do not add anything else."}
    ]
    for shot in few_shot:
        # print(shot)
        context =  shot["context_tweet"]
        text = shot["text"]
        prompt.append({"role": "user", "content": f"News headline:\n{context}\nReader comment:\n{text}\n"})
        prompt.append({"role": "system", "content": generate_response(shot)})
    context = row["context_tweet"]
    text = row["text"]
    prompt.append({"role": "user", "content": f"News headline:\n{context}\nReader comment:\n{text}\n"})
    return prompt


def print_row_as_tsv(row, columns):
    return "\t".join([str(row[column]) for column in columns]) + "\n"

if args.prompt_testing:
    labels = {}
    prediction = {}
    for category in categories:
        labels[category] = []
        prediction[category] = []

    w = open(f"test_results_{args.fewshot_samples}_{args.test_sample_size}.txt", "w")

    for example in sample.to_dict(orient="records"):
            prompt = get_prompt(example, few_shot=FEWSHOT_SAMPLES.to_dict(orient="records"))
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=100,
                temperature=0.0,
                stop=None
            )
            reply = completion.choices[0].message.content
            categories_labeled = reply.split(", ")
            for category in categories:
                labels[category].append(example[category])
                prediction[category].append(1 if category in categories_labeled else 0)
        # break

    for category in categories:
        w.write("---------------------------------")
        w.write(labels[category])
        w.write("---------------------------------")
        w.write(prediction[category])
        if all([elem1 == elem2 for elem1, elem2 in zip(labels[category], prediction[category])]):
            f1 = 1.0
        else:
            f1 = f1_score(labels[category], prediction[category])
        w.write(f"F1 Score for {category}: {f1}")
else:
    for dataset in [to_label_d1, to_label_d2]:
        name = dataset["name"]
        w = open("Datasets/" + name + " - Anotación ChatGPT.tsv", "w")
        for example in dataset["dataset"].to_dict(orient="records"):
            prompt = get_prompt(example, few_shot=FEWSHOT_SAMPLES.to_dict(orient="records"))
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=100,
                temperature=0.0,
                stop=None
            )
            reply = completion.choices[0].message.content
            categories_labeled = reply.split(", ")
            
            for category in categories:
                example[category] = 1 if category in categories_labeled else 0
            
            w.write(print_row_as_tsv(example, list(dataset["dataset"].columns)))
        
        w.close()