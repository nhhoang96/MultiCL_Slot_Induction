
import numpy as np
import random
import torch
import os

def generate_batch(num_sample, batch_size):
    indices = np.arange(num_sample)
    np.random.shuffle(indices)
    batch_index = indices[:batch_size]
    return batch_index

def set_seed(seed_val):
    random.seed(seed_val)
    torch.cuda.cudnn_enabled=False
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.determistic=True
    torch.backends.cudnn.benchmark=False
    os.environ['PYTHONHASHSEED'] = str(seed_val)        
    np.random.seed(seed_val)
    if (torch.cuda.is_available()):
        os.environ['CUDA_LAUNCH_BLOCKING']='1'
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
        os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7,8'
        
    torch.use_deterministic_algorithms(True)




