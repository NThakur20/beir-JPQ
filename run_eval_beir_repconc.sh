set -e
cudaNum=10

dataset=$1 # "nfcorpus" "scifact" "arguana" "scidocs" "fiqa" "trec-covid" ...
M=96 # number of sub-vectors
split=${3:-"test"} # test/dev/train
encode_batch_size=128

echo dataset: $dataset
echo M: $M
echo split: $split

beir_data_root="./data/beir"
output_dir=""
model_root="/home/ukp/thakur/projects/RepCONC/data/passage"
query_encoder="${model_root}/official_query_encoders/m${M}.marcopass.query.encoder"
doc_encoder="${model_root}/official_doc_encoders/m${M}.marcopass.pq.encoder"

for dataset in nfcorpus robust04 trec-news signal1m cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress msmarco bioasq
do
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.eval_beir \
        --dataset $dataset \
        --beir_data_root $beir_data_root \
        --split $split \
        --encode_batch_size $encode_batch_size \
        --query_encoder $query_encoder \
        --doc_encoder $doc_encoder \
        --output_index_path "${output_dir}/index" \
        --output_ranking_path "${output_dir}/${split}-ranking.pickle" \
        --prefix "repconc-m96"
done