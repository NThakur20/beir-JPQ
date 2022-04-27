cudaNum=14

#### PREPROCESSING ####
for dataset in cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress
do
    export PREFIX=gen
    export QRELS_FILE=train.tsv

    ### Convert BEIR dataset to JPQ Friendly ####
    OMP_NUM_THREADS=6 python beir_transformation.py ${dataset} $PREFIX $QRELS_FILE

    # #### PREPROCESSING ####
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.preprocess \
                                    --data_type 1 \
                                    --data_dir /home/ukp/thakur/projects/JPQ/datasets/${dataset} \
                                    --dataset ${dataset}
    
    # #### INIT ####
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.run_init \
        --preprocess_dir /home/ukp/thakur/projects/JPQ/preprocess/${dataset} \
        --model_dir /home/ukp/thakur/projects/JPQ/data/passage/download_dual_encoders/m96/m96.marcopass.doc.encoder \
        --max_doc_length 350 \
        --output_dir /home/ukp/thakur/projects/JPQ/init/${dataset} \
        --subvector_num 96
    
    #### TRAIN ####
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.run_train_qgen \
        --preprocess_dir /home/ukp/thakur/projects/JPQ/preprocess/${dataset} \
        --model_save_dir /home/ukp/thakur/projects/JPQ/final_models/${dataset}/qgen \
        --log_dir /home/ukp/thakur/projects/JPQ/logs/${dataset}/log \
        --init_index_path /home/ukp/thakur/projects/JPQ/init/${dataset}/OPQ96,IVF1,PQ96x8.index \
        --init_model_path /home/ukp/thakur/projects/JPQ/data/passage/download_dual_encoders/star \
        --lambda_cut 200 \
        --centroid_lr 1e-4 \
        --train_batch_size 32 \
        --num_train_epochs 2 \
        --gpu_search \
        --max_seq_length 64
    
    #### Convert ROBERTADOT Models to JPQTower ####
    export METHOD=qgen
    OMP_NUM_THREADS=6 python robertadot_jpq_converter.py ${dataset} $METHOD 96

        #### Evaluation on BEIR Dataset ####
    export beir_data_root=./data/beir
    export query_encoder=/home/ukp/thakur/projects/JPQ/final_models/${dataset}/qgen/jpqtower-query
    export doc_encoder=/home/ukp/thakur/projects/JPQ/final_models/${dataset}/qgen/jpqtower-doc

    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.eval_beir \
        --dataset ${dataset} \
        --beir_data_root $beir_data_root \
        --split test \
        --encode_batch_size 128 \
        --query_encoder $query_encoder \
        --doc_encoder $doc_encoder \
        --prefix "jpq-qgen-32x-compression"
done