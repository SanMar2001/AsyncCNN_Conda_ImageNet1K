import os
import subprocess
from dotenv import load_dotenv

def main():
    load_dotenv()

    device = os.getenv("DEVICE", "CPU").strip().upper()

    if device == "CUDA" or device == "NVIDIA":
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif device == "AMD" or device == "AMD":
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0"
    elif device == "APPLE" or device == "MPS":
        cmd = "pip install torch torchvision torchaudio"
    else:
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    subprocess.run(cmd, shell=True, check=True)

    import torch
    if torch.cuda.is_available():
        print(f'Device = {torch.cuda.get_device_name()}')
    elif torch.backends.mps.is_available():
        print("Device = Apple Silicon")
    else:
        print("Device = CPU")

if __name__ == "__main__":
    main()