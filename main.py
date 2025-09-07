from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import base64

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
import cv2
    
# === CUSTOM PADDING CLASS ===
class CustomPadToSquare1500:
    """
    Custom transformation to pad an image to a square of 1500x1500 pixels.
    If the image is larger, it's centrally cropped. If smaller, it's padded with black (0,0,0) color.
    """
    def __call__(self, img: Image.Image):
        w, h = img.size

        if w > 1500 or h > 1500:
            left = max((w - 1500) // 2, 0)
            top = max((h - 1500) // 2, 0)
            right = left + min(1500, w)
            bottom = top + min(1500, h)
            img = img.crop((left, top, right, bottom))
            w, h = img.size

        pad_left = (1500 - w) // 2
        pad_right = (1500 - w + 1) // 2
        pad_top = (1500 - h) // 2
        pad_bottom = (1500 - h + 1) // 2
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        return TF.pad(img, padding, fill=(0, 0, 0))

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GoogLeNet Model
model = None
try:  
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    num_ftrs_googlenet = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs_googlenet, 2)
    )

    model.load_state_dict(torch.load('best_googlenet_model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("GoogLeNet model loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'best_googlenet_model.pth' file not found. Ensure it's in the same directory.")
    raise HTTPException(status_code=500, detail="AI model not found.")
except Exception as e:
    print(f"ERROR: Could not load the model: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading AI model: {str(e)}")

custom_pad_transform = CustomPadToSquare1500()

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
mean_tensor = torch.tensor(NORM_MEAN).view(3, 1, 1)
std_tensor = torch.tensor(NORM_STD).view(3, 1, 1)

def preprocess_image(image_bytes):
    """
    Preprocesses the input image bytes: opens, converts to RGB,
    applies custom padding, converts to tensor, and normalizes.
    Returns the processed tensor and original image size.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size

        processed_image_pil = custom_pad_transform(image)
        tensor_image = TF.to_tensor(processed_image_pil)
        normalized_tensor = (tensor_image - mean_tensor) / std_tensor

        return normalized_tensor.unsqueeze(0), original_size

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def generate_grad_cam(model, input_tensor, original_image_pil, target_class_idx=None):
    """
    Generates a Grad-CAM heatmap for the given model and input tensor.
    The heatmap highlights regions in the image that are important for the prediction.
    Args:
        model: The PyTorch model.
        input_tensor: The preprocessed image tensor ready for the model.
        original_image_pil: The PIL image before any padding/resizing.
        target_class_idx: The index of the class for which to generate the CAM.
                          If None, the predicted class is used.
    Returns:
        A PIL Image with the Grad-CAM heatmap superimposed on the original image.
    """
    target_layer = model.inception5b

    input_tensor.requires_grad_(True)

    activations = []
    gradients = []

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    hook_handle_activation = target_layer.register_forward_hook(save_activation)
    hook_handle_gradient = target_layer.register_backward_hook(save_gradient)

    output = model(input_tensor)

    if target_class_idx is None:
        target_class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class_idx] = 1
    output.backward(gradient=one_hot_output, retain_graph=True)

    hook_handle_activation.remove()
    hook_handle_gradient.remove()

    guided_gradients = gradients[0].cpu().data.numpy()[0]
    feature_map = activations[0].cpu().data.numpy()[0]

    weights = np.mean(guided_gradients, axis=(1, 2))

    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    cam = np.nan_to_num(cam)

    padded_width, padded_height = input_tensor.shape[3], input_tensor.shape[2]
    cam_resized = cv2.resize(cam, (padded_width, padded_height), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_image_np = np.array(original_image_pil.convert('RGB'))

    heatmap_final = cv2.resize(heatmap, original_image_np.shape[1::-1])

    superimposed_img = heatmap_final * 0.4 + original_image_np * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    result_image_pil = Image.fromarray(superimposed_img)

    return result_image_pil

class PredictionResult(BaseModel):
    """
    Pydantic model for the prediction API response.
    Includes prediction label, confidence, and Base64 encoded Grad-CAM image.
    """
    prediction: str
    confidence: float
    grad_cam_image: str

@app.post("/predict", response_model=PredictionResult)
async def predict_pixel_art(file: UploadFile = File(...)):
    """
    Accepts a pixel art image file and predicts whether it was created
    by AI or a human, using a PyTorch GoogLeNet model.
    Also returns a Grad-CAM visualization.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    image_bytes = await file.read()

    if model is None:
        raise HTTPException(status_code=500, detail="AI model was not loaded correctly.")

    try:
        original_image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        processed_image_tensor, original_size = preprocess_image(image_bytes)
        processed_image_tensor = processed_image_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = model(processed_image_tensor)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            idx_to_class = {0: "AI", 1: "Human"}
            result_label = idx_to_class[predicted_idx.item()]

        model.eval()
        grad_cam_image_pil = generate_grad_cam(model, processed_image_tensor, original_image_pil, predicted_idx.item())

        buffered = io.BytesIO()
        grad_cam_image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "prediction": result_label,
            "confidence": confidence.item(),
            "grad_cam_image": img_str
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during prediction or Grad-CAM generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
