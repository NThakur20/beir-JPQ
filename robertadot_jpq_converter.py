import faiss
import torch
import sys, os
from transformers import RobertaConfig


from jpq.model import RobertaDot, JPQTower

dataset = sys.argv[1]
method = sys.argv[2] if len(sys.argv) > 2 else "qgen"
M = int(sys.argv[3]) if len(sys.argv) > 2 else 96
for model_type in ["query", "doc"]:
    if M == 96:
        if os.path.isdir(f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{method}/epoch-2"):
            if model_type == "query":
                robertadot_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{method}/epoch-2"
                init_index_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{method}/epoch-2/OPQ96,IVF1,PQ96x8.index"
        else:
            if model_type == "query":
                robertadot_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{method}/epoch-1"
                init_index_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{method}/epoch-1/OPQ96,IVF1,PQ96x8.index"

        if model_type == "doc":
            robertadot_path = f"/home/ukp/thakur/projects/JPQ/data/passage/download_dual_encoders/star"
            init_index_path = f"/home/ukp/thakur/projects/JPQ/init/{dataset}/OPQ96,IVF1,PQ96x8.index"
        output_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{method}/jpqtower-{model_type}"
    
    elif M != 96:
        if os.path.isdir(f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{M}/{method}/epoch-2"):
            if model_type == "query":
                robertadot_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{M}/{method}/epoch-2"
                init_index_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{M}/{method}/epoch-2/OPQ{M},IVF1,PQ{M}x8.index"
        else:
            if model_type == "query":
                robertadot_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{M}/{method}/epoch-1"
                init_index_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{M}/{method}/epoch-1/OPQ{M},IVF1,PQ{M}x8.index"

        if model_type == "doc":
            robertadot_path = f"/home/ukp/thakur/projects/JPQ/data/passage/download_dual_encoders/star"
            init_index_path = f"/home/ukp/thakur/projects/JPQ/init/{dataset}/{M}/OPQ{M},IVF1,PQ{M}x8.index"
        output_path = f"/home/ukp/thakur/projects/JPQ/final_models/{dataset}/{M}/{method}/jpqtower-{model_type}"

    opq_index = faiss.read_index(init_index_path)

    vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))            
    assert isinstance(vt, faiss.LinearTransform)
    opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

    ivf_index = faiss.downcast_index(opq_index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)

    centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
    centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
    centroid_embeds += coarse_embeds.reshape(ivf_index.pq.M, -1, ivf_index.pq.dsub)
    coarse_embeds[:] = 0   

    config = RobertaConfig.from_pretrained(robertadot_path)
    config.name_or_path = output_path
    config.MCQ_M, config.MCQ_K = ivf_index.pq.M, ivf_index.pq.ksub
    
    model = JPQTower.from_pretrained(robertadot_path, config=config)

    with torch.no_grad():
        model.centroids.copy_(torch.from_numpy(centroid_embeds))
        model.rotation.copy_(torch.from_numpy(opq_transform))

    model.save_pretrained(output_path)