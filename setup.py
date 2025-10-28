import subprocess

def main():
    print("Select the type of GPU in case of have it:\n")
    print("1. NVIDIA")
    print("2. AMD")
    print("3. Apple Silicon")
    print("4. Only CPU\n")

    choice = input("Select your device: ")

    if choice == "1":
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif choice == "2":
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0"
    elif choice == "3":
        cmd = "pip install torch torchvision torchaudio"
    else:
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
