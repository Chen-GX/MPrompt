import numpy as np 
import random 
import torch 
import os 

def set_seed(seed: int = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def save_model(model, tokenizer, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    #如果我们有一个分布式模型，只保存封装的模型
    #它包装在PyTorch DistributedDataParallel或DataParallel中
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)
    # torch.save(model_to_save.state_dict(), osp.join(path, 'pytorch_model.bin'))
    # model_to_save.config.to_json_file(osp.join(path, 'config.json'))
    # tokenizer.save_vocabulary(path)
    tokenizer.save_pretrained(path)

