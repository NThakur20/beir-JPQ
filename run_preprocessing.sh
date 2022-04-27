cudaNum=7

CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.preprocess \
                                --data_type 1 \
                                --data_dir /home/ukp/thakur/projects/JPQ/datasets/fiqa \
                                --dataset fiqa 
