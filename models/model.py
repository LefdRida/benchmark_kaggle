"""This module contains functions to extract features from images, audio, and text using various models."""
from urllib.parse import urlparse
from .models_utils import *
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import ViTModel, ViTImageProcessor
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import requests
from io import BytesIO

from PIL import Image
import io
def load_image(img_struct):
    return Image.open(io.BytesIO(img_struct["bytes"]))

# from aim.torch.models import AIMForImageClassification
# from aim.torch.data import val_transforms
from transformers import AutoImageProcessor, ViTMAEForPreTraining

def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    

cache_dir = "/home/rida.lefdali/lustre/scalableml-um6p-st-sccs-10v5rwpbsmu/lefdali-lustre/embedding_model"
def cosplace_img(img_files: list, batch_size: int = 32) -> np.ndarray:
    """Extract image features using CosPlace model specifically trained for the Pittsburgh dataset.

    Args:
        img_files: list of image files
        batch_size: batch size
    Returns:
        image features
    """
    transforms_list = []
    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transforms_list)
    model = torch.hub.load(
        "gmberton/cosplace",
        "get_trained_model",
        backbone="ResNet50",
        fc_output_dim=2048,
    )
    model = model.cuda()
    total_params = sum(param.numel() for param in model.parameters())
    img_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            for img_file in img_files[i : i + batch_size]:
                image = Image.open(img_file).convert("RGB")
                image = transform(image).unsqueeze(0)
                images.append(image)
            batch = torch.cat(images, dim=0).cuda()
            outputs = model(batch)
            img_embeddings.append(outputs.detach().cpu().numpy())
    return np.concatenate(img_embeddings, axis=0), total_params



def ibot(img_files: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    assert model_variant in ['ibot-base', 'ibot-large'], f"Model Variant ({model_variant}) Is Not Supported"
    if model_variant == "ibot-base":
        model = vit_base()
    elif model_variant == "ibot-base":
        model = vit_large()
    else:
        raise f"Model Variant ({model_variant}) Is Not Supported"
    
    load_pretrained_weights_ibot(model, model_name = model_variant)
    model = model.cuda().half()
    total_params = sum(param.numel() for param in model.parameters())
    model.dtype = torch.float16
    image_processor = CustomImageProcessor()
    img_embeddings = []
    print(f"number of params {total_params}")
    with torch.no_grad():
        
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            if isinstance(img_files[i]["bytes"], (bytes, bytearray)):
                for img_file in img_files[i : i + batch_size]:    
                    image = Image.open(BytesIO(img_file["bytes"])).convert("RGB")
                    images.append(image)
            elif isinstance(img_files[i], str):
                for img_file in img_files[i : i + batch_size]:
                    if is_url(img_file):
                        response = requests.get(img_file)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(img_file).convert("RGB")
                    images.append(image)
            inputs = image_processor(images, return_tensors="pt").to("cuda", dtype=model.dtype)
            outputs = model(inputs)
            img_embeddings.append(outputs.detach().cpu().numpy())
    return np.concatenate(img_embeddings, axis=0), total_params
    

def dinov1_vitb16(img_files: list[str], batch_size: int = 32, model_variant=None) -> np.ndarray:
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    model = model.cuda()
    total_params = sum(param.numel() for param in model.parameters())
    image_processor = CustomImageProcessor()
    img_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            for img_file in img_files[i : i + batch_size]:
                image = Image.open(img_file).convert("RGB")
                images.append(image)
            inputs = image_processor(images, return_tensors="pt").to("cuda")
            outputs = model(inputs)
            img_embeddings.append(outputs.detach().cpu().numpy())
    return np.concatenate(img_embeddings, axis=0), total_params

def google_vit(img_files: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    """Extract image features using Vision Transformer model.

    Args:
        img_files: list of image files
        batch_size: batch size
    Returns:
        image features
    """
    assert model_variant in ['vit-large-patch16-224-in21k', 'vit-base-patch16-224-in21k', 'vit-huge-patch14-224-in21k'], f"Model Variant ({model_variant}) Is Not Supported"
    print(f"Loading {model_variant.upper()}")
    processor = ViTImageProcessor.from_pretrained(f"google/{model_variant}")
    model = ViTModel.from_pretrained(f"google/{model_variant}")
    model = model.cuda()
    total_params = sum(param.numel() for param in model.parameters())
    img_embeddings = []
    print(f"number of params {total_params}")
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            for img_file in img_files[i : i + batch_size]:
                image = Image.open(img_file).convert("RGB")
                images.append(image)
            batch = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**batch)
            image_features = outputs.last_hidden_state
            image_features = image_features.mean(dim=1)
            img_embeddings.append(image_features.detach().cpu().numpy())

    return np.concatenate(img_embeddings, axis=0), total_params

def dinov3(img_files: list[str], batch_size: int = 50, model_variant=None):
    """Extract image features using DINO model.

    Args:
        img_files: list of image files
        batch_size: batch size
    Returns:
        image features
    """
    assert model_variant in ['dinov3-vit7b16-pretrain-lvd1689m', 'dinov3-vith16plus-pretrain-lvd1689m', 'dinov3-vitl16-pretrain-lvd1689m', 'dinov3-vitb16-pretrain-lvd1689m', 'dinov3-vits16-pretrain-lvd1689m'], f"Model Variant ({model_variant}) Is Not Supported"
    
    print(f"Loading {model_variant.upper()}")
    processor = AutoImageProcessor.from_pretrained(f"facebook/{model_variant}")
    model = AutoModel.from_pretrained(f"facebook/{model_variant}", cache_dir=cache_dir)
    model = model.cuda()
    total_params = sum(param.numel() for param in model.parameters())
    img_embeddings = []
    print(f"number of params {total_params}")
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            if isinstance(img_files[i]["bytes"], (bytes, bytearray)):
                for img_file in img_files[i : i + batch_size]:    
                    image = Image.open(BytesIO(img_file["bytes"])).convert("RGB")
                    images.append(image)
            elif isinstance(img_files[i], str):
                for img_file in img_files[i : i + batch_size]:
                    if is_url(img_file):
                        response = requests.get(img_file)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(img_file).convert('RGB')  
                    images.append(image)
            elif isinstance(img_files[i], Image.Image):
                images = img_files[i : i + batch_size]
            batch = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**batch)
            image_features = outputs.pooler_output
            img_embeddings.append(image_features.detach().cpu().numpy())

    return np.concatenate(img_embeddings, axis=0), total_params



def dinov2(img_files: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    """Extract image features using DINO model.

    Args:
        img_files: list of image files
        batch_size: batch size
    Returns:
        image features
    """
    assert model_variant in ['dinov2-giant', 'dinov2-base', 'dinov2-small', 'dinov2-large'], f"Model Variant ({model_variant}) Is Not Supported"
    
    print(f"Loading {model_variant.upper()}")
    processor = AutoImageProcessor.from_pretrained(f"facebook/{model_variant}")
    model = AutoModel.from_pretrained(f"facebook/{model_variant}", cache_dir=cache_dir)
    model = model.cuda()
    total_params = sum(param.numel() for param in model.parameters())
    img_embeddings = []
    print(f"number of params {total_params}")
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            if isinstance(img_files[i]["bytes"], (bytes, bytearray)):
                for img_file in img_files[i : i + batch_size]:    
                    image = Image.open(BytesIO(img_file["bytes"])).convert("RGB")
                    images.append(image)
            elif isinstance(img_files[i], str):
                for img_file in img_files[i : i + batch_size]:
                    if is_url(img_file):
                        response = requests.get(img_file)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(img_file).convert("RGB")
                    images.append(image)
            elif isinstance(img_files[i], Image.Image):
                images = img_files[i : i + batch_size]
            batch = processor(images, return_tensors="pt").to("cuda")
            outputs = model(**batch)
            image_features = outputs.last_hidden_state
            image_features = image_features.mean(dim=1)
            img_embeddings.append(image_features.detach().cpu().numpy())
    
    return np.concatenate(img_embeddings, axis=0), total_params



def ijepa(img_files: list[str], batch_size: int = 50, model_variant=None):
    print(model_variant)
    assert model_variant in ['ijepa_vith14_22k', 'ijepa_vitg16_22k'], f"Model Variant ({model_variant}) Is Not Supported"
    processor = AutoProcessor.from_pretrained(f"facebook/{model_variant}")
    model = AutoModel.from_pretrained(f"facebook/{model_variant}", device_map="cuda", cache_dir=cache_dir)
    total_params = sum(param.numel() for param in model.parameters())
    img_embeddings = []
    print(f"number of params {total_params}")
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            if isinstance(img_files[i]["bytes"], (bytes, bytearray)):
                for img_file in img_files[i : i + batch_size]:    
                    image = Image.open(BytesIO(img_file["bytes"])).convert("RGB")
                    images.append(image)
            elif isinstance(img_files[i], str):
                for img_file in img_files[i : i + batch_size]:
                    if is_url(img_file):
                        response = requests.get(img_file)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(img_file).convert("RGB")
                    images.append(image)
            inputs = processor(images,  return_tensors="pt").to('cuda')
            outputs = model(**inputs)
            outputs = outputs.last_hidden_state.mean(dim=1)
            img_embeddings.append(outputs.detach().cpu().numpy())
    return np.concatenate(img_embeddings, axis=0), total_params


# def aim(img_files: list[str], batch_size: int = 50, model_variant=None):
#     assert model_variant in ['aim-600M', 'aim-1B', 'aim-3B'], f"Model Variant ({model_variant}) Is Not Supported"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = AIMForImageClassification.from_pretrained(f"apple/{model_variant}").to(device)
#     transform = val_transforms()
#     total_params = sum(param.numel() for param in model.parameters())
#     img_embeddings = []
#     print(f"number of params {total_params}")
#     with torch.no_grad():
#         for i in tqdm(range(0, len(img_files), batch_size)):
#             images = []
#             for img_file in img_files[i : i + batch_size]:
#                 if is_url(img_file):
#                     response = requests.get(img_file)
#                     image = Image.open(BytesIO(response.content)).convert("RGB")
#                 else:
#                     image = Image.open(img_file).convert("RGB")
#                 images.append(image)
#             inputs = transform(images).unsqueeze(0).to(device)
#             _, features = model(inputs)
#             img_embeddings.append(features.detach().cpu().numpy())
#     return np.concatenate(img_embeddings, axis=0), total_params

def mae(img_files: list[str], batch_size: int = 50, model_variant=None):
    assert model_variant in ['vit-mae-base', 'vit-mae-large'], f"Model Variant ({model_variant}) Is Not Supported"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViTImageProcessor.from_pretrained(f"facebook/{model_variant}")
    model = ViTMAEForPreTraining.from_pretrained(f"facebook/{model_variant}", attn_implementation="sdpa").to(device)
    total_params = sum(param.numel() for param in model.parameters())
    img_embeddings = []
    print(f"number of params {total_params}")
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size)):
            images = []
            if isinstance(img_files[i]["bytes"], (bytes, bytearray)):
                for img_file in img_files[i : i + batch_size]:    
                    image = Image.open(BytesIO(img_file["bytes"])).convert("RGB")
                    images.append(image)
            elif isinstance(img_files[i], str):
                for img_file in img_files[i : i + batch_size]:
                    if is_url(img_file):
                        response = requests.get(img_file)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        image = Image.open(img_file).convert("RGB")
                    images.append(image)
            inputs = processor(images,  return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            outputs = outputs.last_hidden_state.mean(dim=1)
            img_embeddings.append(outputs.detach().cpu().numpy())
    return np.concatenate(img_embeddings, axis=0), total_params


def sentence_t5(text: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    """Extract text features using GTR model.

    Args:
        text: list of text
    Returns:
        text features
    """
    assert model_variant in ['sentence-t5-xxl', 'sentence-t5-xl', 'sentence-t5-large', 'sentence-t5-base'], f"Model Variant ({model_variant}) Is Not Supported"
    model = SentenceTransformer(f"sentence-transformers/{model_variant}", cache_folder=cache_dir)
    model = model.cuda()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"number of params {total_params}")
    return model.encode(text), total_params

def gtr_t5(text: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    """Extract text features using GTR model.

    Args:
        text: list of text
    Returns:
        text features
    """
    assert model_variant in ['gtr-t5-xxl', 'gtr-t5-xl', 'gtr-t5-large', 'gtr-t5-base'], f"Model Variant ({model_variant}) Is Not Supported"
    model = SentenceTransformer(f"sentence-transformers/{model_variant}", cache_folder=cache_dir)
    model = model.cuda()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"number of params {total_params}")
    return model.encode(text), total_params

def all_mpnet_base_v2(text: list[str],  batch_size: int = 50, model_variant=None):
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    pooling = mean_pooling
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=device, cache_dir=cache_dir).half()
    total_params = sum(param.numel() for param in model.parameters())
    text_features = []
    print(f"number of params {total_params}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size] 
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)
            # Compute token embeddings
            model_output = model(**inputs)
            sentence_embeddings = pooling(model_output, inputs['attention_mask'])
            text_features.append(sentence_embeddings.detach().cpu().numpy())
    return np.concatenate(text_features, axis=0), total_params
    
def alibaba_gte_en_v1_5(text: list[str],  batch_size: int = 50, model_variant=None):
    assert model_variant in ['gte-base-en-v1.5', 'gte-large-en-v1.5'], f"Model Variant ({model_variant}) Is Not Supported"
    model_name = f"Alibaba-NLP/{model_variant}"
    pooling = bos_pooling
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=device, cache_dir=cache_dir).half()
    total_params = sum(param.numel() for param in model.parameters())
    text_features = []
    print(f"number of params {total_params}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size] 
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)
            # Compute token embeddings
            model_output = model(**inputs)
            sentence_embeddings = pooling(model_output, inputs['attention_mask'])
            text_features.append(sentence_embeddings.detach().cpu().numpy())
    return np.concatenate(text_features, axis=0), total_params

    
def baai_bge_en_v1_5(text: list[str],  batch_size: int = 50, model_variant=None) -> np.ndarray:
    # Load model from HuggingFace Hub
    assert model_variant in ['bge-base-en-v1.5', 'bge-large-en-v1.5'], f"Model Variant ({model_variant}) Is Not Supported"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = f"BAAI/{model_variant}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map=device, cache_dir=cache_dir)
    model.eval()
    total_params = sum(param.numel() for param in model.parameters())
    # Tokenize sentences
    text_features = []
    print(f"number of params {total_params}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size] 
            encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
            # Compute token embeddings
            model_output = model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            text_features.append(sentence_embeddings.detach().cpu().numpy())
    return np.concatenate(text_features, axis=0), total_params


def gte_qwen2_1_5B_instruct(text: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
    pooling = last_token_pool
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map=device, cache_dir=cache_dir).half()
    total_params = sum(param.numel() for param in model.parameters())
    inputs = tokenizer(text, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)

    with torch.autocast(device_type=device.type, dtype=model.dtype):  # or bfloat16
        model_output = model(**inputs)
    sentence_embeddings = pooling(model_output, inputs['attention_mask'])

    return sentence_embeddings.detach().cpu().numpy(), total_params


def infloat_e5(text: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    # Load model from HuggingFace Hub
    assert model_variant in ['e5-small-v2', 'e5-base-v2', 'e5-large-v2'], f"Model Variant ({model_variant}) Is Not Supported"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(f'intfloat/{model_variant}', cache_dir=cache_dir)
    model = AutoModel.from_pretrained(f'intfloat/{model_variant}', device_map=device, cache_dir=cache_dir)
    model.eval()
    total_params = sum(param.numel() for param in model.parameters())
    # Tokenize sentences
    text_features = []
    print(f"number of params {total_params}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size] 
            encoded_input = tokenizer(batch, max_length=1024, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**encoded_input)
            embeddings = average_pool_infloat_e5(outputs.last_hidden_state, encoded_input['attention_mask'])
            text_features.append(embeddings.detach().cpu().numpy())
            
    return np.concatenate(text_features, axis=0), total_params
    


def nv_embed(text: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    assert model_variant in ['NV-Embed-v2'], f"Model Variant ({model_variant}) Is Not Supported"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    max_length = 32768
    batch_size = 50
    model = AutoModel.from_pretrained(f'nvidia/{model_variant}', trust_remote_code=True, device_map=device, cache_dir=cache_dir)
    model.eval()
    total_params = sum(param.numel() for param in model.parameters())
    text_features = []
    print(f"number of params {total_params}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size] 
            outputs = model.encode(batch, instruction="Given a web search query, retrieve relevant passages that answer the query.\n")
            text_features.append(outputs.detach().cpu().numpy())
            
    return np.concatenate(text_features, axis=0), total_params


def qwen3(text: list[str], batch_size: int = 50, model_variant=None) -> np.ndarray:
    assert model_variant in ['Qwen3-Embedding-8B', 'Qwen3-Embedding-4B', 'Qwen3-Embedding-0.6B'], f"Model Variant ({model_variant}) Is Not Supported"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    max_length = 32768
    tokenizer = AutoTokenizer.from_pretrained(f'Qwen/{model_variant}', padding_side='left')
    model = AutoModel.from_pretrained(f'Qwen/{model_variant}', trust_remote_code=True, device_map=device, cache_dir=cache_dir)
    model.eval()
    total_params = sum(param.numel() for param in model.parameters())
    text_features = []
    print(f"number of params {total_params}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(text), batch_size)):
            batch = text[i : i + batch_size] 
            batch_dict = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict.to(device)
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            text_features.append(embeddings.detach().cpu().numpy())
            
    return np.concatenate(text_features, axis=0), total_params
    


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
            images = []
            if isinstance(img_files[i], dict) and isinstance(img_files[i].get("bytes", None), (bytes, bytearray)):
                print(type(img_files[i]))
                for img_file in img_files[i : i + batch_size]:    
                    image = Image.open(BytesIO(img_file['bytes'])).convert("RGB")
                    images.append(preprocess(image))
            if isinstance(img_files[i], (str, Path)):
                for img_file in img_files[i : i + batch_size]:
                    image = Image.open(img_file).convert("RGB")
                    images.append(preprocess(image))
            elif isinstance(img_files[i], Image.Image):
                for img_file in img_files[i : i + batch_size]:
                    image = img_file.convert("RGB")
                    images.append(preprocess(image))
            else:
                print(img_files[i])
                raise ValueError("Unsupported image format")

            batch = torch.stack(images).to(device)

            with torch.amp.autocast(device_type=device):
                features = model.encode_image(batch)

            features = features / features.norm(dim=-1, keepdim=True)
            img_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(img_embeddings, axis=0)
    return embeddings, total_params
