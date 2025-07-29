import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the embedding model
MODEL_NAME = "intfloat/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)