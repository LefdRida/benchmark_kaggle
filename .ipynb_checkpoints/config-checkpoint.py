config = {
    "tasks": ["imagenet1k"], # dataset_name
    "method_name": "csa",  # Method to use: "asif", "csa", or "cka"
    "csa":{
        "sim_dim": 700,
    },
    "asif":{
        "non_zeros": 10,
    },
    "retrieval":{
        "topk": 5,
        "num_gt": 1,
    },
    "support_embeddings": None,
    "imagenet1k": {
        "root": "/home/rida.lefdali/work/ImageNet/val",
        "loc_val_solution": "/home/rida.lefdali/work/ImageNet/LOC_val_solution.csv",
        "loc_synset_mapping": "/home/rida.lefdali/work/ImageNet/LOC_synset_mapping.txt",
        "img_encoder": "dinov2-large", 
        "text_encoder": "gtr-t5-large", 
        "hf_img_embedding_name": "ImageNet_img_embed_dinov2-large.pkl", 
        "hf_text_embedding_name": "ImageNet_text_embed_gtr-t5-large.pkl", 
        "hf_repo_id": "ridalefdali/ImageNet_embeddings", 
        "train_test_ratio": 0.8,
        "seed": 42,
        "split": "large",
        "generate_embedding": False,
        "batch_size": 50,
        "metatask": "classification", # only "classification"
    },
    "flickr30k": {
        "dataset_path": "",
        "img_encoder": "", 
        "text_encoder": "", 
        "hf_img_embedding_name": "", 
        "hf_text_embedding_name": "", 
        "hf_repo_id": "", 
        "train_test_ratio": 0.8,
        "seed": 42,
        "split": "large",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": True,
        "batch_size": 50,
        "metatask": "retrieval", # only "retrieval"
    },
    "mscoco": {
        "data_path": "",
        "img_encoder": "", 
        "text_encoder": "", 
        "hf_img_embedding_name": "", 
        "hf_text_embedding_name": "", 
        "hf_repo_id": "", 
        "train_test_ratio": 0.8,
        "seed": 42,
        "split": "large",
        "num_caption_per_image": 5,
        "num_image_per_caption": 1,
        "generate_embedding": True,
        "batch_size": 50,
        "metatask": "retrieval", # "classification" or "retrieval"
    }
}