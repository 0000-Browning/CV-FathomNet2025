import torch

def test_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Please check your GPU setup.")

if __name__ == "__main__":
    test_cuda()