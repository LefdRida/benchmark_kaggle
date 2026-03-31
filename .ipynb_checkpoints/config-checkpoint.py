config = {
    "tasks": ["mscoco"], # dataset_name
    "methods": ["csa", "asif"],  # Method to use: "asif", "csa", or "cka"
    "csa":{
        "sim_dim": 700,
    },
    "asif":{
        "non_zeros": 800,
    },
    "retrieval":{
        "topk": 5,
        "num_gt": 5,
    },
    "classification":{
    },
    "support_embeddings": None,

    "imagenet1k": {
        "root": "/home/rida.lefdali/work/ImageNet/val",
        "loc_val_solution": "/home/rida.lefdali/work/ImageNet/LOC_val_solution.csv",
        "loc_synset_mapping": "/home/rida.lefdali/work/ImageNet/LOC_synset_mapping.txt",
        "hf_img_embedding_name": "ImageNet_img_embed_dinov2-giant.pkl", 
        "hf_text_embedding_name": "ImageNet_text_embed_gtr-t5-large.pkl", 
        "hf_repo_id": "ridalefdali/ImageNet_embeddings", 
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "generate_embedding": False,
        "metatask": "classification", # only "classification"
    },
    "flickr30k": {
        "dataset_path": "/home/rida.lefdali/work/dataset/flickr30k",
        "hf_img_embedding_name": "flickr30k_dinov2_dinov2-giant_image_embeddings.pkl", 
        "hf_text_embedding_name": "flickr30k_gtr_t5_gtr-t5-large_text_embeddings.pkl", 
        "hf_repo_id": "ridalefdali/flickr30k_embeddings", 
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": False,
        "metatask": "retrieval", # only "retrieval"
    },
    "mscoco": {
        "data_path": "/home/rida.lefdali/work/coco2017_dataset/coco2017",
        "hf_img_embedding_name": "mscoco_dinov2_dinov2-giant_image_embeddings.pkl", 
        "hf_text_embedding_name": "mscoco_gtr_t5_gtr-t5-large_text_embeddings.pkl", 
        "hf_repo_id": "ridalefdali/mscoco_classification_embeddings", #"ridalefdali/mscoco_classification_embeddings"
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": False,
        "metatask": "classification", # "classification" or  "retrieval"
    },
    "image_embedding_models": ["dinov2"],
    "text_embedding_models": ["all_mpnet_base_v2", "gtr_t5", "sentence_t5"]#"alibaba_gte_en_v1_5", "baai_bge_en_v1_5"],
    "embedding_model": {
        "img_encoder": "dinov2", 
        "text_encoder": "gtr_t5", 
        "image_model_variant": "dinov2-giant",
        "text_model_variant": "gtr-t5-large",
        "batch_size": 128,
    }
}