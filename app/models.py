from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class MultimodalEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('clip-ViT-B-32')
        
    def encode_text(self, text: str):
        return self.model.encode(text).tolist()
    
    def encode_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode(image).tolist()
    
    def encode_batch(self, items):
        # items can be mix of text and images
        return self.model.encode(items)