import numpy as np
import torch
import open_clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple


def clip_text(
    text: List[str],
    batch_size: int = 32,
    model_variant: str = "ViT-L-14",
) -> Tuple[np.ndarray, int]:

    model, _, _ = open_clip.create_model_and_transforms(
        model_variant,
        pretrained="openai",
    )

    tokenizer = open_clip.get_tokenizer(model_variant)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of CLIP parameters: {total_params}")

    text_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i: i + batch_size]
            batch = tokenizer(batch).to(device)

            with torch.amp.autocast(device_type=device):
                features = model.encode_text(batch)

            features = features / features.norm(dim=-1, keepdim=True)
            text_features.append(features.cpu().numpy())

    embeddings = np.concatenate(text_features, axis=0)
    return embeddings, total_params



def clip_img(
    img_files: List,
    batch_size: int = 32,
    model_variant: str = "ViT-L-14",
) -> Tuple[np.ndarray, int]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_variant,
        pretrained="openai",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of CLIP parameters: {total_params}")

    img_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            batch = []

            for img_file in img_files[i: i + batch_size]:
                if isinstance(img_file, (str, Path)):
                    image = Image.open(img_file).convert("RGB")
                elif isinstance(img_file, Image.Image):
                    image = img_file.convert("RGB")
                else:
                    raise ValueError("Unsupported image format")

                batch.append(preprocess(image))

            batch = torch.stack(batch).to(device)

            with torch.amp.autocast(device_type=device):
                features = model.encode_image(batch)

            features = features / features.norm(dim=-1, keepdim=True)
            img_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(img_embeddings, axis=0)
    return embeddings, total_params