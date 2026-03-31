from data.get_data_embedding import get_data_embedding
from models import get_image_embedding_model_variant, get_text_embedding_model_variant
from config import config
from omegaconf import OmegaConf


if __name__ == "__main__":

    config = OmegaConf.create(config)
    # 1. Run Embeddings
    for task in config.tasks:
        for img_embedding_model in config.image_embedding_models:
            OmegaConf.update(config, "embedding_model.img_encoder", img_embedding_model)
            for txt_embedding_model in config.text_embedding_models:
                OmegaConf.update(config, "embedding_model.text_encoder", txt_embedding_model)
                for img_variant in get_image_embedding_model_variant(img_embedding_model):
                    OmegaConf.update(config, "embedding_model.image_model_variant", img_variant)
                    for txt_variant in get_text_embedding_model_variant(txt_embedding_model):
                        OmegaConf.update(config, "embedding_model.text_model_variant", txt_variant)
                        print(f'Embedding {task} dataset with {config["embedding_model"]} model')
                        get_data_embedding(task, config[task.lower()], config["embedding_model"])