# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 as cv
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from models.model import SAAN

mean = [0.485, 0.456, 0.406]  
std = [0.229, 0.224, 0.225]

def image_loader(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"No image in {image_path}")
        return None, image_path
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img = transform(img)
    return img, image_path

def predict_images(model, image_paths, device):
    predictions = []
    for img_path in image_paths:
        img, img_path = image_loader(img_path)
        if img is not None:
            with torch.no_grad():
                img = img.unsqueeze(0).to(device)
                predicted_label = model(img)
                prediction = predicted_label.squeeze().cpu().numpy()
                predictions.append(prediction * 10)
                print(f"{img_path}: {prediction * 10}")
        else:
            predictions.append(None)
            print(f"Skipping {img_path} due to loading error.")
    return predictions

pred_img = [f"predict/{i}.jpg" for i in range(1, 51)]
pred_img_filter = [f"predict_filter/{i}tr.jpg" for i in range(1, 51)]

checkpoint_path = "checkpoint/BAID/model_best.pth"
resnet_checkpoint_path = 'checkpoint/ResNet_Pretrain/epoch_99.pth'

if not os.path.exists(resnet_checkpoint_path):
    raise FileNotFoundError(f"File {resnet_checkpoint_path} does not exist.")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = SAAN(num_classes=1)
model = model.to(device)

map_location = torch.device('cpu') if device == "cpu" else None
model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
model.eval()

predictions = predict_images(model, pred_img, device)
predictions_filter = predict_images(model, pred_img_filter, device)

print("Predictions:", predictions)
print("Predictions Filter:", predictions_filter)

# %%
matrix = np.column_stack((predictions, predictions_filter))

# Check the combined matrix
print("Combined Matrix:")
print(matrix)

# %%
csv_file_path = 'info.csv'
data = pd.read_csv(csv_file_path)

score_dict = {row['New Name']: row['Score'] for _, row in data.iterrows()}

score_matrix = np.array([score_dict[i] for i in range(1, 51)]).reshape(50, 1)

print("Score Matrix:")
print(score_matrix)

# %%
A = matrix
b = score_matrix
A_transpose = A.T
A_transpose_A = np.dot(A_transpose, A)
A_transpose_A_inv = np.linalg.inv(A_transpose_A)
A_transpose_b = np.dot(A_transpose, b)
W = np.dot(A_transpose_A_inv, A_transpose_b)

print("W Matrix:")
print(W)

# %%
