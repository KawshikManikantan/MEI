from flask import Flask, render_template, request, jsonify, Markup
import os
import jsonlines
from utils.get_processed_dataset import *
from process_text import *
from omegaconf import OmegaConf

app = Flask(__name__)

# Assuming datasets and splits are known and static
DATASETS = ["litbank", "fantasy"]
SPLITS = ["dev", "test"]


@app.route("/")
def index():
    return render_template("index.html", datasets=DATASETS, splits=SPLITS)


@app.route("/get_documents", methods=["POST"])
def get_documents():
    dataset = request.json["dataset"]
    split = request.json["split"]
    documents_path = f"../mei-data/raw_data/{dataset}/text/{split}"
    try:
        documents = os.listdir(documents_path)
        documents = [doc for doc in documents if doc.endswith(".txt")]
    except FileNotFoundError:
        documents = []
    return jsonify(documents)


@app.route("/load_document", methods=["POST"])
def load_document():
    dataset = request.json["dataset"]
    split = request.json["split"]
    document_txt = request.json["document"]
    document = document_txt.split(".")[0]

    document_path = f"../mei-data/raw_data/{dataset}/text/{split}/{document_txt}"
    result_dict_path = (
        f"../results/final1/gpt/h2s/{dataset}/{split}/full/desc/result.jsonl"
    )
    # result_dict_path = f"../results/final1/lf/met/sd-gen/setup4/run5/{dataset}/{split}/result.jsonl"

    with open(document_path, "r") as file:
        doc_text = file.read()

    result_dict = {}
    with jsonlines.open(result_dict_path) as reader:
        for obj in reader:
            if obj["doc_key"] == document:
                result_dict = obj
                break
    # print(result_dict)

    dataset_config = OmegaConf.load(f"utils/datasets.yaml")
    doc_addr = dataset_config[dataset][f"{split}_file"]

    doc_tsv_addr = dataset_config[dataset]["tsv"]
    ## For now we are not using SOTA
    is_sota = False
    # is_sota = True
    # sota_str = "_sota"
    sota_str = ""  # "_sota" if is_sota else ""
    head = False  ##

    doc_me = dataset_config[dataset][f"{split}_me{sota_str}"]

    docs = get_coref_docs(doc_addr)
    docs_processed = get_processed_dataset(docs, doc_tsv_addr, head=head)
    # print(docs_processed.keys())
    major_entities = get_major_entities(doc_me)

    colored_content1 = gold_full(
        doc_text,
        results_doc=result_dict,
        doc_processed=docs_processed[document],
        major_entities=major_entities[document],
    )
    # colored_content2 = gold_head(doc_text,results_doc=result_dict,doc_processed=docs_processed[document],major_entities=major_entities[document])
    # colored_content3 = predicted_head(doc_text,results_doc=result_dict,doc_processed=docs_processed[document],major_entities=major_entities[document])
    colored_content4 = predicted_full(
        doc_text,
        results_doc=result_dict,
        doc_processed=docs_processed[document],
        major_entities=major_entities[document],
    )

    return render_template(
        "text-box.html",
        content1=colored_content1,
        content2="",
        content3="",
        content4=colored_content4,
    )
    # return render_template("text-box.html",content1=colored_content1, content2=colored_content2, content3="colored_content3", content4=colored_content4)
    # return jsonify(content=content)


if __name__ == "__main__":
    app.run(debug=True)
