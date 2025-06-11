import torch.nn as nn
import timm

def load_model():
    # Inisialisasi model sesuai arsitektur training
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)  # 2 kelas: organik dan daur ulang
    return model