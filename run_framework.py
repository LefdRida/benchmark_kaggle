import numpy as np
from benchmark import MMA_Benchmark
from tasks.classification import ClassificationTask
from tasks.retrieval import RetrievalTask
from methods.asif import ASIFMethod
from methods.csa import CSAMethod
from models.model_wrappers import HFModelWrapper
from datasets.loader import load_dataset_task
from config import config
from omegaconf import OmegaConf



config = OmegaConf.create(config)

def test_framework():
    
    # 2. Create Tasks
    tasks = []
    for task in config.tasks:
        tasks.append(load_dataset_task(task, config[task.lower()]))
    support_embeddings = None
    # 3. Create Benchmark
    benchmark = MMA_Benchmark(tasks=tasks)
    
    # 4. Create Method
    asif = ASIFMethod(non_zeros=config.asif.non_zeros)
    
    # 5. Run Benchmark (Model is dummy here as we pre-encoded)
    # In a real scenario, you'd pass a model to encode raw data if needed
    results = benchmark.run(method=asif, model=None, support_embeddings=support_embeddings)
    
    print("ASIF Results:", results)

    # Test CSA
    csa = CSAMethod(sim_dim=config.csa.sim_dim)
    results_csa = benchmark.run(method=csa, model=None, support_embeddings=support_embeddings)
    print("CSA Results:", results_csa)

if __name__ == "__main__":
    test_framework()
