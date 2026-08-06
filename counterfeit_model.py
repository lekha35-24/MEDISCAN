"""import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "models/best_model.pth"  # <-- matches train_model.py now

def load_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_counterfeit(model, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "Fake" if pred.item() == 0 else "Real"
    confidence = conf.item()
    return label, confidence

def grad_cam(model, image_tensor, target_class=None):
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]
    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    handle1.remove()
    handle2.remove()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Use all available CPU cores for torch ops (helps both model inference
# and EasyOCR, which is also torch-backed) — safe, no behavior change.
torch.set_num_threads(os.cpu_count())
cv2.setUseOptimized(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "models/best_model.pth"  # <-- matches train_model.py now


def load_model():
    # weights=None instead of ResNet18_Weights.IMAGENET1K_V1:
    # we overwrite every layer (including fc) with our own trained
    # checkpoint immediately after, so downloading/initializing the
    # ImageNet weights here was pure wasted time. Final loaded model
    # is identical either way.
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    try:
        # weights_only=True is faster and safer to load when your
        # torch version supports it; falls back cleanly if not.
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_counterfeit(model, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # inference_mode is a faster variant of no_grad — skips autograd
    # version-counter bookkeeping since we never need gradients here.
    with torch.inference_mode():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "Fake" if pred.item() == 0 else "Real"
    confidence = conf.item()
    return label, confidence


def grad_cam(model, image_tensor, target_class=None):
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]
    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    handle1.remove()
    handle2.remove()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lets cuDNN pick the fastest conv algorithms for your fixed 224x224 input
# size. Only affects GPU runtime, zero effect on outputs/logic — safe no-op
# on CPU.
torch.backends.cudnn.benchmark = True

CHECKPOINT_PATH = "models/best_model.pth"  # <-- matches train_model.py now


def load_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # weights_only=True is safe here because train_model.py saves a plain
    # state_dict (an OrderedDict of tensors) — nothing else is pickled in
    # that file, so restricting the unpickler doesn't change what loads,
    # it just skips PyTorch's slower full-unpickle safety path.
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_counterfeit(model, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # inference_mode is a stricter/faster version of no_grad for pure
    # inference (no .backward() anywhere in this path) — same numbers out,
    # less autograd bookkeeping overhead.
    with torch.inference_mode():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "Fake" if pred.item() == 0 else "Real"
    confidence = conf.item()
    return label, confidence


def grad_cam(model, image_tensor, target_class=None):
    # Left as torch.no_grad-free / normal autograd — grad_cam NEEDS
    # gradients (loss.backward()), so it can't use inference_mode.
    # Unchanged from your version.
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]
    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    handle1.remove()
    handle2.remove()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam