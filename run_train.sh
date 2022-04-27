cudaNum=0

for dataset in fiqa
do
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.run_train_qgen \
        --preprocess_dir /home/ukp/thakur/projects/JPQ/preprocess/${dataset} \
        --model_save_dir /home/ukp/thakur/projects/JPQ/final_models/${dataset}/qgen \
        --log_dir /home/ukp/thakur/projects/JPQ/logs/${dataset}/log \
        --init_index_path /home/ukp/thakur/projects/JPQ/init/${dataset}/OPQ96,IVF1,PQ96x8.index \
        --init_model_path /home/ukp/thakur/projects/JPQ/data/passage/download_dual_encoders/star \
        --lambda_cut 200 \
        --centroid_lr 1e-4 \
        --train_batch_size 32 \
        --num_train_epochs 2
done