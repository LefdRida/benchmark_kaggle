config = {
    "tasks": ["imagenet1k", "flickr30k"], # dataset_name
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
        "root": "",
        "loc_val_solution": "",
        "loc_synset_mapping": "",
        "img_encoder": "", 
        "text_encoder": "", 
        "hf_img_embedding_name": "", 
        "hf_text_embedding_name": "", 
        "hf_repo_id": "", 
        "train_test_ratio": 0.8,
        "seed": 42,
        "split": "large",
        "generate_embedding": True,
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