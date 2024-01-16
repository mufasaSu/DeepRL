import torch
print("Is MPS available:", "Yes" if torch.backends.mps.is_available() else "No")
print(torch.__version__)
use_gpu = True
if torch.cuda.is_available() and use_gpu:
    device = torch.device("cuda:" + str(gpu_id))
    print("Using GPU id {}".format(gpu_id))



def init_gpu(use_gpu=True,): #gpu_id=0):
    global device
    if torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")
        print("Using GPU id Mac")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

init_gpu(use_gpu=True)