from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

import pathlib, os, csv, random
import sys
import logging
import json
from tqdm import tqdm

random.seed(42)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = sys.argv[1]
prefix = sys.argv[2]
qrels_file = sys.argv[3]
hard_negatives = sys.argv[4] if len(sys.argv) > 4 else None

print("Coverting {}...".format(dataset))
data_path = "/home/ukp/thakur/projects/sbert_retriever/datasets-new/{}".format(dataset)

output_dir = "/home/ukp/thakur/projects/JPQ/datasets/{}".format(dataset)
os.makedirs(output_dir, exist_ok=True)

query_sum = 0
doc_sum = 0

#### Provide the data_path where nfcorpus has been downloaded and unzipped
data_path = "/home/ukp/thakur/projects/sbert_retriever/datasets-new/{}".format(dataset)
corpus, queries, _ = GenericDataLoader(data_path, prefix=prefix).load(split="train")
qrels = {}
qrels_filepath = os.path.join(data_path, prefix + "-qrels", qrels_file)

reader = csv.reader(open(qrels_filepath, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
next(reader)
for id, row in enumerate(reader):
    query_id, corpus_id, score = row[0], row[1], int(row[2])
    
    if query_id not in qrels:
        qrels[query_id] = {corpus_id: score}
    else:
        qrels[query_id][corpus_id] = score

queries = {k : queries[k] for k in qrels.keys()}
corpus_ids = list(corpus)
query_ids = list(queries)

doc_map, query_map = {}, {}

for idx, corpus_id in enumerate(corpus_ids): 
    doc_map[corpus_id] = idx

for idx, query_id in enumerate(query_ids):
    query_map[query_id] = idx

print("Writing Corpus to file")
with open(os.path.join(output_dir, "collection.tsv"), 'w', encoding="utf-8") as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for doc_id in tqdm(corpus_ids, total=len(corpus_ids)):
        doc = corpus[doc_id]
        doc_id_new = doc_map[doc_id]
        writer.writerow([doc_id_new, (doc.get("title", "").replace("\r", " ").replace("\t", " ").replace("\n", " ") + " " + doc.get("text", "").replace("\r", " ").replace("\t", " ").replace("\n", " ")).strip()])

print("Writing Queries to file")
with open(os.path.join(output_dir, "queries.tsv"), 'w', encoding="utf-8") as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for qid, query in tqdm(queries.items(), total=len(queries)):
        qid_new = query_map[qid]
        writer.writerow([qid_new, query])

print("Writing Qrels to file")
with open(os.path.join(output_dir, "qrels.train.tsv"), 'w', encoding="utf-8") as fIn:
    writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for qid, docs in tqdm(qrels.items(), total=len(qrels)):
        for doc_id, score in docs.items():
            qid_new = query_map[qid]
            doc_id_new = doc_map[doc_id]
            writer.writerow([qid_new, 0, doc_id_new, score])

if hard_negatives:
    result_json = []
    print("Writing Hard Negatives to file")
    with open(os.path.join(data_path, hard_negatives), encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                qid_new = query_map[line["qid"]]
                line_new = {
                'qid': query_map[line["qid"]],
                'pos': [doc_map[line["pos"][0]]],
                'neg': {k: [doc_map[doc_id] for doc_id in v] for k, v in line["neg"].items()}}
                result_json.append(line_new)
    
    with open(os.path.join(output_dir, "hard_negatives.jsonl"), 'w') as fIn:
            for line in tqdm(result_json, total=len(result_json)):
                fIn.write(json.dumps(line) + '\n')
