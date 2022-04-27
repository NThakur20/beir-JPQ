import argparse
import logging
import random, os, pathlib
import pickle, faiss
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from jpq.model import JPQDualEncoder
from jpq.model import DenseRetrievalJPQSearch as DRJS

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--beir_data_root", type=str, required=True)
parser.add_argument("--query_encoder", type=str, required=True)
parser.add_argument("--doc_encoder", type=str, required=True)
parser.add_argument("--split", type=str, default='test')
parser.add_argument("--encode_batch_size", type=int, default=64)
parser.add_argument("--output_index_path", type=str, default=None)
parser.add_argument("--output_ranking_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default="jpq")
args = parser.parse_args()

#### Download scifact.zip dataset and unzip the dataset
dataset = args.dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# data_path = util.download_and_unzip(url, args.beir_data_root)
data_path = "/home/ukp/thakur/projects/sbert_retriever/datasets-new/{}".format(dataset)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)

#### Load pre-computed index
if args.output_index_path is not None and os.path.isfile(args.output_index_path):
    corpus_index = faiss.read_index(args.output_index_path)
else:
    corpus_index = None

print(corpus_index)

#### Load the RepCONC model and retrieve using dot-similarity
model = DRJS(JPQDualEncoder((args.query_encoder, args.doc_encoder),), batch_size=args.encode_batch_size, corpus_index=corpus_index)
retriever = EvaluateRetrieval(model, score_function="dot") # or "dot" for dot-product
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")

# if args.output_index_path is not None:
#     os.makedirs(os.path.dirname(args.output_index_path), exist_ok=True)
#     faiss.write_index(model.corpus_index, args.output_index_path)

if args.output_ranking_path:
    output_dir = os.path.join("/home/ukp/thakur/projects/beir/examples/retrieval", "output", dataset, args.output_ranking_path)
else:
    output_dir = os.path.join("/home/ukp/thakur/projects/beir/examples/retrieval", "output", dataset)

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "{}_results.txt".format(args.prefix))

fOut = open(output_path, 'w')
fOut.write("\t".join(["Queries: {}".format(len(queries)), "Corpus: {}".format(len(corpus))]))
fOut.write("\n")

for k in _map.keys():
    fOut.write("{}: {:.5f}".format(k, _map[k]))
    fOut.write("\n")

fOut.write("\n")
for k in ndcg.keys():
    fOut.write("{}: {:.5f}".format(k, ndcg[k]))
    fOut.write("\n")

fOut.write("\n")
for k in recall.keys():
    fOut.write("{}: {:.5f}".format(k, recall[k]))
    fOut.write("\n")

fOut.write("\n")
for k in precision.keys():
    fOut.write("{}: {:.5f}".format(k, precision[k]))
    fOut.write("\n")

fOut.write("\n")
for k in mrr.keys():
    fOut.write("{}: {:.5f}".format(k, mrr[k]))
    fOut.write("\n")

fOut.write("\n")
for k in recall_cap.keys():
    fOut.write("{}: {:.5f}".format(k, recall_cap[k]))
    fOut.write("\n")

fOut.write("Examples:")
if len(queries) > 50:
    for _ in range(50):
        query_id, results_ = random.choice(list(results.items()))
        fOut.write("\n")
        fOut.write("\n")
        fOut.write("Query: {}".format(queries[query_id]))
        gold_passages = qrels[query_id]
        results_sorted = sorted(results_.items(), key=lambda item: item[1], reverse=True)
        for idx in range(20):
            fOut.write("\n")
            corpus_id = results_sorted[idx][0]
            gold = gold_passages[corpus_id] if corpus_id in gold_passages else "No"
            fOut.write("{}: {} ({}) {} - {}".format(idx+1, corpus_id, gold, corpus[corpus_id].get("title"), corpus[corpus_id].get("text")))

else:
    for query_id, results_ in results.items():
        fOut.write("\n")
        fOut.write("\n")
        fOut.write("Query: {}".format(queries[query_id]))
        gold_passages = qrels[query_id]
        results_sorted = sorted(results_.items(), key=lambda item: item[1], reverse=True)
        for idx in range(20):
            fOut.write("\n")
            corpus_id = results_sorted[idx][0]
            gold = gold_passages[corpus_id] if corpus_id in gold_passages else "No"
            fOut.write("{}: {} ({}) {} - {}".format(idx+1, corpus_id, gold, corpus[corpus_id].get("title"), corpus[corpus_id].get("text")))
