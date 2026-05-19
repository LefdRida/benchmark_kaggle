import json
from typing import List, Union, Dict, Any
from base.base import AbsTask, AbsModel, AbsMethod
import json

class MMA_Benchmark:
    """Main Orchestrator for the MMA Benchmark"""

    def __init__(self, tasks: List[AbsTask], methods: Dict[str, AbsMethod], config: Dict[str, Any]):
        self.tasks = tasks
        self.methods = methods
        self.config = config

    def run(self, model: AbsModel, support_embeddings: Dict[str, Any] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Run all tasks in the benchmark for a given method and model."""
        results = {}
        diagnostic_results = {}
        for task in self.tasks:
            print(f"Running task: {task.name} ({task.task_type})")
            for method_name, method_class in self.methods.items():
                exp_name = f"{task.name}_{method_name}_{self.config.retrieval.experiment_name}_{self.config.retrieval.n_clusters}"
                print(exp_name)
                method = method_class(**self.config[method_name])
                print(f"Running method: {method.name}")
                results[exp_name], diagnostic_results[exp_name] = task.run(method, support_embeddings=support_embeddings, **kwargs)
                # with open("cka_results_retrieval.json", "w") as f:
                #     json.dump(results, f)
        return results, diagnostic_results

    def add_task(self, task: AbsTask):
        """Add a new task to the benchmark."""
        self.tasks.append(task)
