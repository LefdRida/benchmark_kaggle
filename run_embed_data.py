from data.get_data_embedding import get_data_embedding
from config import config
from omegaconf import OmegaConf


if __name__ == "__main__":

    config = OmegaConf.create(config)
    # 1. Run Embeddings
    for task in config.tasks:
        get_data_embedding(task, config[task.lower()])