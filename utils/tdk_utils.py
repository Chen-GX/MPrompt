import os
import torch
import logging
import torch.nn as nn
from pathlib import Path
from transformers import AutoConfig
from torch.utils.data import DataLoader
from dataset_class.my_dataset import tdk_QA_dataset
logger = logging.getLogger(__name__)

def load_tdkdata(args, tokenizer):
    print("load datasets ...")  
    # file_dir, filename, tokenizer, source_len, target_len, data_split='Train'
    train_ds = tdk_QA_dataset(args, tokenizer, data_split='train')    
    val_ds = tdk_QA_dataset(args, tokenizer, data_split='val')
    test_ds = tdk_QA_dataset(args, tokenizer, data_split='test')
    
    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
    }
    
    val_params = {
        "batch_size": args.val_batch_size,
        "shuffle": False,
    }
    
    training_loader = DataLoader(train_ds, **train_params)
    val_loader = DataLoader(val_ds, **val_params)  
    test_loader = DataLoader(test_ds, **val_params) 
    return training_loader, val_loader, test_loader



# TODO: domain prompt 
def get_config(args):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    config.update({
        'domain_size': args.domain_size,
        'use_domain': args.use_domain,
        'use_enc_dec': args.use_enc_dec,  # 是否约束enc和dec之间的prompt
        'domain_loss_name': args.domain_loss_name,  # 损失名
        'loss_sample_n': args.loss_sample_n,
        'use_encoder_prompt': True  if args.use_domain and args.use_encoder_prompt else False,
        'use_decoder_prompt': True  if args.use_domain and args.use_decoder_prompt else False,
        'init_from_vocab': args.init_from_vocab,
        'pre_seq_len': args.n_prompt_tokens,
    })
    return config

# TODO: 
def save_tdk_prompt(model, path, filename = "tdk_prompt.model"):
    Path(path).mkdir(parents=True, exist_ok=True)

    save_state = {}
    print("Model's state_dict:")

    for param_name in model.state_dict():
        if 'pretrain_model' in param_name and 'knowledge_pretrain_model' not in param_name:
            continue  # 如果没有第二个条件，会跳过knowledge pretrain model中的所有参数
        # 'knowledge_prompt_encoder.knowledge_pretrain_model.shared.weight'
        if 'knowledge_pretrain_model' in param_name:
            if "prompt_encoder_list" in param_name or "prompt_decoder_list" in param_name or 'prompt_tokens' in param_name:
                pass
            else:
                continue
        # print(param_name) 
        # # save_state.update({param_name:torch.ones((model.state_dict()[param_name].size()))})
        save_state.update({param_name:model.state_dict()[param_name]})
        # # print(param_name, "\t", model.state_dict()[param_name].size())

    torch.save(save_state, os.path.join(path, filename))
    print(f"Saved prefix param at: {os.path.join(path, filename)}")


# TODO: 确定待训练的参数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    base_model_num = sum(p.numel() for p in net.pretrain_model.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f'参数量 | Total: {total_num} | base model: {base_model_num} | Trainable: {trainable_num}')
    # return {'Total': total_num, 'Trainable': trainable_num}


def get_group_parameters(args, model):
    task_names = []
    task_params = []
    knowlege_names = []
    knowlege_params = []
    domain_names = []
    domain_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'knowledge_embed' in name:
                knowlege_names.append(name)
                knowlege_params.append(param)
            elif 'prompt_encoder_list' in name or 'prompt_decoder_list' in name:
                domain_names.append(name)
                domain_params.append(param)
            else:
                task_names.append(name)
                task_params.append(param)
    # NOTO: 这里knowledge部分只有embedding层是独立参数的，其他的和task一样（指MLP映射部分）
    param_group = []
    if len(task_params) > 0:
        param_group.append({'params': task_params, 'lr': args.lr})

    if len(domain_params) > 0:
        param_group.append({'params': domain_params, 'lr': args.domain_lr})

    if len(knowlege_params) > 0:
        param_group.append({'params': knowlege_params, 'lr': args.knowledge_lr})

    return param_group



