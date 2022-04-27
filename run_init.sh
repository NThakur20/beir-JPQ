cudaNum=1

for dataset in fiqa
do
  CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m jpq.run_init \
    --preprocess_dir /home/ukp/thakur/projects/JPQ/preprocess/${dataset} \
    --model_dir /home/ukp/thakur/projects/JPQ/data/passage/download_dual_encoders/m96/m96.marcopass.doc.encoder \
    --max_doc_length 350 \
    --output_dir /home/ukp/thakur/projects/JPQ/init/${dataset} \
    --subvector_num 96
done