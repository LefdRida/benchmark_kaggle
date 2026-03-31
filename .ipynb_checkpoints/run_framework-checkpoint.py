import numpy as np
from benchmark import MMA_Benchmark
from metatasks.classification import ClassificationTask
from metatasks.retrieval import RetrievalTask
from methods import get_method_class
from data.loader import load_dataset_metatask
from config import config
from omegaconf import OmegaConf

from models import get_image_embedding_model_variant, get_text_embedding_model_variant
import json

    
config = OmegaConf.create(config)

def test_framework():
    results_logs = {}
    # 1. Run Embeddings
     img_embedding_model_name = []
    # for img_embedding_model in config.image_embedding_models:
    #     for img_variant in get_image_embedding_model_variant(img_embedding_model):
    #         img_embedding_model_name.append(f"{img_embedding_model}_{img_variant}")

    # txt_embedding_model_name = []
    # for txt_embedding_model in config.text_embedding_models:
    #     for txt_variant in get_text_embedding_model_variant(txt_embedding_model):
    #         txt_embedding_model_name.append(f"{txt_embedding_model}_{txt_variant}")

    #for img_embedding_model_name in img_embedding_model_name:
    #    for txt_embedding_model_name in txt_embedding_model_name:
    #        OmegaConf.update(config, "mscoco.hf_img_embedding_name", f"mscoco_{img_embedding_model_name}_image_embeddings.pkl")
    #        OmegaConf.update(config, "mscoco.hf_text_embedding_name", f"mscoco_{txt_embedding_model_name}_text_embeddings.pkl")
    #        print(f"Running {img_embedding_model_name} with {txt_embedding_model_name}")
    # 2. Create Tasks
    tasks = []
    for task in config.tasks:
        tasks.append(load_dataset_metatask(task, config))
    support_embeddings = None
    
    print("Tasks loaded")
    # 4. Create Method dynamically from config
    # Example: config.method_name = "asif" or "csa"
    methods = {}
    for method_name in config.methods:
        MethodClass = get_method_class(method_name)
        methods[method_name] = MethodClass
    print("Methods loaded")
    # 3. Create Benchmark
    benchmark = MMA_Benchmark(tasks=tasks, methods=methods, config=config)
    print("Benchmark created")
    # 5. Run Benchmark
    print("Running Benchmark")
    results = benchmark.run(model=None, support_embeddings=support_embeddings)
    print(f"Results:", results)
    #results_logs[f"{img_embedding_model_name}_{txt_embedding_model_name}"] = results
    #with open("results.json", "w") as f:
    #    json.dump(results_logs, f)
            

if __name__ == "__main__":
    test_framework()
