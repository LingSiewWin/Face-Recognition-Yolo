from facenet_pytorch import InceptionResnetV1
import cv2
import torch
import numpy as np

face_encoder = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(face):
    face = cv2.resize(face, (160, 160))  
    face = torch.tensor(face).float().permute(2, 0, 1).unsqueeze(0)
    embedding = face_encoder(face).detach().numpy()
    return embedding
