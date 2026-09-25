import os

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

IMG_SIZE = 224
MODEL_PATH = "approach3_resnet_finetuned.pth"
label_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"Model weights not found: {path}\n"
            "Download approach3_resnet_finetuned.pth from the Google Drive link in "
            "the README and place it in the project root, then run this again."
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(label_names))

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


model = load_model(MODEL_PATH)

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_waste(img):
    img_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = F.softmax(model(img_tensor), dim=1)[0]
    return {label_names[i]: float(probabilities[i]) for i in range(len(label_names))}


demo = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="Upload a waste item photo"),
    outputs=gr.Label(num_top_classes=6, label="Prediction"),
    title="Waste Classification for Recycling",
    description="Upload a photo of a waste item to see the model's prediction.",
    flagging_mode="never"
)

demo.launch()
