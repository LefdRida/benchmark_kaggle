config = {
    "tasks": [ "imagenet1k", "places365", "mscoco"], # dataset_name "imagenet1k",
    "methods": ["knn"],  # Method to use: "asif", "csa", or "cka" "csa", 
    "csa":{
        "sim_dim": 700,
    },
    "asif":{
        "non_zeros": 800,
    },
    "knn":{
        "num_classes": 1000,
    },
    "retrieval":{
        "topk": 5,
        "num_gt": 5,
    },
    "classification":{
    },
    "embedding_space_analysis":{
        
    },
    "support_embeddings": None,

    "imagenet1k": {
        "root": "/home/rida.lefdali/work/dataset/imagenet1k/val",
        "loc_val_solution": "/kaggle/input/datasets/odddatascience/imagenet1k/LOC_val_solution.csv",
        "loc_synset_mapping": "/kaggle/input/datasets/odddatascience/imagenet1k/LOC_synset_mapping.txt",
        "hf_img_embedding_name": "ImageNet_img_embed_dinov2-giant.pkl", 
        "hf_text_embedding_name": "ImageNet_text_embed_gtr-t5-large.pkl", 
        "hf_repo_id": "ridalefdali/imagenet1k_classification_embeddings", 
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "original_ds_split": "val",
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
        "original_ds_split": "all",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": False,
        "metatask": "retrieval", # only "retrieval"
    },
    "mscoco": {
        "data_path": "/home/rida.lefdali/work/dataset/coco2017",
        "hf_img_embedding_name": "mscoco_dinov2_dinov2-giant_image_embeddings.pkl", 
        "hf_text_embedding_name": "mscoco_gtr_t5_gtr-t5-large_text_embeddings.pkl", 
        "hf_repo_id": "ridalefdali/mscoco_classification_embeddings", #"ridalefdali/mscoco_classification_embeddings"
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "original_ds_split": "train",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": False,
        "metatask": "classification", # "classification" or  "retrieval"
    },
    "places365": {
        "root": "/home/rida.lefdali/work/dataset/places365_standard/train",
        "filelist_places": "/kaggle/input/datasets/odddatascience/places365/places365_train_standard.txt",
        "categories_places": "/kaggle/input/datasets/odddatascience/places365/categories_places365.txt",
        "hf_img_embedding_name": "places365_dinov2_dinov2-giant_image_embeddings.pkl", 
        "hf_text_embedding_name": "places365_gtr_t5_gtr-t5-large_text_embeddings.pkl", 
        "hf_repo_id": "ridalefdali/places365_classification_embeddings", #"ridalefdali/mscoco_classification_embeddings"
        "train_test_ratio": 0.7,
        "seed": 42,
        "split": "large",
        "original_ds_split": "train",
        "generate_embedding": False,
        "metatask": "classification", # "classification" or  "retrieval"
    },
    "image_embedding_models": ["dinov2"],
    "text_embedding_models": ["gtr_t5"], #"alibaba_gte_en_v1_5", "baai_bge_en_v1_5"], , "sentence_t5" "all_mpnet_base_v2",
    "embedding_model": {
        "img_encoder": "dinov2", 
        "text_encoder": "gtr_t5", 
        "image_model_variant": "dinov2-giant",
        "text_model_variant": "gtr-t5-large",
        "batch_size": 128,
    }
}