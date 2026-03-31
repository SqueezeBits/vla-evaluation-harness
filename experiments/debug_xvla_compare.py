"""Compare our xvla.py inference vs official deploy.py inference.
Run with: uv run /tmp/debug_xvla_compare.py
"""

# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.44,<=4.51.3",
#     "numpy>=1.24",
#     "pillow>=9.0",
#     "opencv-python-headless",
#     "fastapi",
#     "json-numpy",
#     "uvicorn",
#     "einops",
#     "timm",
# ]
# ///
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

MODEL_PATH = "2toINF/X-VLA-WidowX"
IMAGE_PATH = "/tmp/simpler_widowx_frame0.png"

# Load model (same as our xvla.py)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
if hasattr(config, "florence_config"):
    config.florence_config._attn_implementation_internal = "eager"
model = AutoModel.from_pretrained(
    MODEL_PATH,
    config=config,
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.float32,
)
model.to(device="cuda:0", dtype=torch.float32).eval()

# Load image
pil_img = Image.open(IMAGE_PATH)
print(f"Image: {pil_img.size}, mode={pil_img.mode}")

# Build proprio (official way)
ee_pos = np.array([0.29206935, -0.00609147, 0.13532102], dtype=np.float32)
proprio_np = np.zeros(20, dtype=np.float32)
proprio_np[:3] = ee_pos
proprio_np[3:10] = [1, 0, 0, 1, 0, 0, 0]

instruction = "put the spoon on the towel"
device = torch.device("cuda:0")

# === METHOD 1: Official deploy.py way ===
inputs1 = processor([pil_img], instruction)
inputs1 = {
    k: v.to(device=device, dtype=torch.float32)
    if isinstance(v, torch.Tensor) and v.is_floating_point()
    else v.to(device=device)
    if isinstance(v, torch.Tensor)
    else v
    for k, v in inputs1.items()
}
proprio1 = torch.tensor(proprio_np, dtype=torch.float32, device=device).unsqueeze(0)
domain_id1 = torch.tensor([0], dtype=torch.long, device=device)

with torch.no_grad():
    actions1 = model.generate_actions(**inputs1, proprio=proprio1, domain_id=domain_id1, steps=10)
raw1 = actions1[0].cpu().numpy()

# === METHOD 2: Our xvla.py way ===
inputs2 = processor(images=[pil_img], language_instruction=instruction)
inputs2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs2.items()}
proprio2 = torch.tensor(proprio_np, dtype=torch.float32, device=device).unsqueeze(0)
domain_id2 = torch.tensor([0], dtype=torch.long, device=device)

with torch.no_grad():
    actions2 = model.generate_actions(**inputs2, proprio=proprio2, domain_id=domain_id2, steps=10)
raw2 = actions2[0].cpu().numpy()

print("\n=== RAW 20D ACTION COMPARISON (first action) ===")
print(f"Official: {raw1[0, :10].round(4)}")
print(f"Ours:     {raw2[0, :10].round(4)}")
print(f"Max diff: {np.abs(raw1 - raw2).max():.8f}")
print(f"Pos diff: {np.abs(raw1[0, :3] - raw2[0, :3]).max():.8f}")
print(f"Rot diff: {np.abs(raw1[0, 3:9] - raw2[0, 3:9]).max():.8f}")
print(f"Grip diff: {np.abs(raw1[0, 9] - raw2[0, 9]):.8f}")
