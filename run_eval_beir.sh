set -e

cudaNum=8
dataset=$1 # "nfcorpus" "scifact" "arguana" "scidocs" "fiqa" "trec-covid" ...
M=96 # number of sub-vectors
split=${3:-"test"} # test/dev/train
encode_batch_size=128

echo dataset: $dataset
echo M: $M
echo split: $split

beir_data_root="./data/beir"
output_dir="./output/"
model_root="./data/passage/download_dual_encoders"

for dataset in nq
do
    for M in 96
    do
    export METHOD=gpl
    OMP_NUM_THREADS=6 python robertadot_jpq_converter.py ${dataset} $METHOD ${M}

    export beir_data_root=./data/beir
    export query_encoder=/home/ukp/thakur/projects/JPQ/final_models/${dataset}/gpl/jpqtower-query
    export doc_encoder=/home/ukp/thakur/projects/JPQ/final_models/${dataset}/gpl/jpqtower-doc
    # export output_ranking_path=jpq-diff-compression

    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.eval_beir \
        --dataset $dataset \
        --beir_data_root $beir_data_root \
        --split $split \
        --encode_batch_size $encode_batch_size \
        --query_encoder $query_encoder \
        --doc_encoder $doc_encoder \
        --prefix "jpq-gpl-compression" 
    done
done



# for dataset in cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress
# do
#     export beir_data_root=./data/beir
#     export query_encoder=/home/ukp/thakur/projects/JPQ/final_models/cqadupstack-all/qgen/jpqtower-query
#     export doc_encoder=/home/ukp/thakur/projects/JPQ/final_models/cqadupstack-all/qgen/jpqtower-doc
#     export output_ranking_path=jpq-diff-compression

#     CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.eval_beir \
#         --dataset $dataset \
#         --beir_data_root $beir_data_root \
#         --split $split \
#         --encode_batch_size $encode_batch_size \
#         --query_encoder $query_encoder \
#         --doc_encoder $doc_encoder \
#         --prefix "jpq-qgen-compression" 
# done