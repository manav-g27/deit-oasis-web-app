# app.py
# One-file Gradio app for Swin-B (patch4, window7) classification with robust safe loading on PyTorch 2.6+
# - Broad allowlist for weights_only=True (timm + torch.nn classes)
# - Optional fallback to full unpickling if TRUST_CHECKPOINT=True (only if you trust the file)
# - Edit MODEL_NAME/IMG_SIZE/NUM_CLASSES/CLASS_NAMES to match your training

import os
import torch
import gradio as gr
import timm
from PIL import Image
from torchvision import transforms

# =============== USER CONFIG ===============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "alz_vit.pt")
MODEL_NAME = "deit_base_patch16_224"  # *_384 if trained at 384
IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ["Mild Dementia","Moderate Dementia","Non Demented","Very mild Dementia"]
# Set True ONLY if you fully trust the checkpoint source to allow full unpickling
TRUST_CHECKPOINT = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============== PREPROCESS ===============
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# =============== MODEL BUILD ===============
def build_model():
    return timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

model = build_model().to(DEVICE)

# =============== SAFE ALLOWLIST ===============
def allowlist_required_classes():
    from torch.serialization import add_safe_globals

    # Core torch.nn classes frequently referenced
    try:
        import torch.nn as nn
        add_safe_globals([
            nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d,
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.Identity, nn.ReLU, nn.GELU, nn.SiLU, nn.Softmax, nn.Sigmoid, nn.Tanh,
            nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AvgPool2d, nn.MaxPool2d,
            nn.Sequential, nn.ModuleList, nn.Parameter
        ])
    except Exception:
        pass

    # Common timm layers used by Swin
    try:
        from timm.models.swin_transformer import SwinTransformer
        add_safe_globals([SwinTransformer])
    except Exception:
        pass

    try:
        from timm.models.swin_transformer_v2 import SwinTransformerV2
        add_safe_globals([SwinTransformerV2])
    except Exception:
        pass

    try:
        from timm.layers.patch_embed import PatchEmbed
        add_safe_globals([PatchEmbed])
    except Exception:
        pass

    try:
        from timm.layers.mlp import Mlp
        add_safe_globals([Mlp])
    except Exception:
        pass

    try:
        from timm.layers.drop import DropPath
        add_safe_globals([DropPath])
    except Exception:
        pass

    # Some timm norm/act and attention utilities
    try:
        from timm.layers.norm_act import LayerNorm
        add_safe_globals([LayerNorm])
    except Exception:
        pass

    try:
        # WindowAttention lives with Swin in some timm versions
        from timm.models.swin_transformer import WindowAttention as SwinWindowAttention
        add_safe_globals([SwinWindowAttention])
    except Exception:
        pass

    try:
        # Fallback in other versions
        from timm.layers.attention import Attention
        add_safe_globals([Attention])
    except Exception:
        pass

    # VisionTransformer sometimes appears in mixed checkpoints
    try:
        from timm.models.vision_transformer import VisionTransformer
        add_safe_globals([VisionTransformer])
    except Exception:
        pass

# =============== LOAD CHECKPOINT ===============
def load_checkpoint(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Try safe load with expanded allowlist
    try:
        allowlist_required_classes()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not TRUST_CHECKPOINT:
            raise RuntimeError(
                "Safe load failed and TRUST_CHECKPOINT is False.\n"
                f"{e}\n\n"
                "Option 1: Copy the class path shown in the error (e.g., torch.nn.modules.conv.Conv2d or a timm class)\n"
                "and add it to allowlist_required_classes() using add_safe_globals([...]).\n"
                "Option 2: Set TRUST_CHECKPOINT=True at the top to allow full unpickling (ONLY if you trust the file)."
            )
        # Fallback (potentially unsafe)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # If a full model object was saved
    if isinstance(state, torch.nn.Module):
        mdl = state.to(DEVICE).eval()
        return mdl, [], []

    # If wrapped like {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)}")

    # Strip 'module.' prefixes
    clean_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    model.eval().to(DEVICE)
    return model, missing, unexpected

model, missing_keys, unexpected_keys = load_checkpoint(model, MODEL_PATH)
if missing_keys or unexpected_keys:
    print("Notice: state_dict differences detected.")
    if missing_keys:
        print("Missing keys (first 30):", missing_keys[:30], "..." if len(missing_keys) > 30 else "")
    if unexpected_keys:
        print("Unexpected keys (first 30):", unexpected_keys[:30], "..." if len(unexpected_keys) > 30 else "")

# =============== INFERENCE ===============
@torch.no_grad()
def predict_pil(pil_img: Image.Image):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    if len(CLASS_NAMES) == len(probs):
        top_label = CLASS_NAMES[top_idx]
        prob_map = { CLASS_NAMES[i]: float(p) for i, p in enumerate(probs) }
    else:
        top_label = f"class_{top_idx}"
        prob_map = { f"class_{i}": float(p) for i, p in enumerate(probs) }
    return top_label, prob_map

def gradio_infer(img):
    return predict_pil(img)

# =============== GRADIO UI ===============
demo = gr.Interface(
    fn=gradio_infer,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=[
        gr.Label(label="Predicted class"),
        gr.Label(num_top_classes=len(CLASS_NAMES), label="Class probabilities"),
    ],
    title="Alzheimer's Image Classification (DEIT)",
    description="""
Upload a brain MRI image to classify Alzheimer's disease stage using a trained DeiT (Vision Transformer) model.

⚠️ <b>Disclaimer:</b> This application is intended solely for <b>research and educational purposes</b>. It <b>must not be used for clinical diagnosis, medical decision-making, or patient care</b>. Predictions generated by this model should always be interpreted and validated by qualified healthcare professionals.
"""
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, share=False)
