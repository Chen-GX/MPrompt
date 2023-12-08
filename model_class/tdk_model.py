#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 原始版本，只在encoder加，有self attention和MLP两种模式
import sys
import torch
from torch import nn
import logging
from transformers import AutoTokenizer
from .modeling_t5 import T5ForConditionalGeneration
from .knowledge_prompt import knowledge_prompt

logger = logging.getLogger(__name__)

class T5tdkForConditionalGeneration(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        # self.config = get_config(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, use_fast=True, local_files_only=True)
        self.pretrain_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, local_files_only=True)
        self.config = self.pretrain_model.config

        if isinstance(self.pretrain_model, (T5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model  # 模型embd长度
            self.match_n_embd = self.config.d_kv
        else:
            raise ValueError("Other models are not supported yet!")
        
        # prompt related.

        # initial task-specific prompt
        if args.use_task:
            self.task_sequence_length = args.task_sequence_length
            self.task_mid_dim = args.task_mid_dim
            self.init_task_prompt()

        # initial knowledge prompt
        if args.use_knowledge:
            self.knowledge_prompt_encoder = knowledge_prompt(args)  # 这里的参数knowledge_prompt里面冻结
        

        # logger.info("task-specific prompt sequence length is {}.".format(self.task_sequence_length))
        
        self.dropout = nn.Dropout(args.prompt_dropout)

        if self.args.freeze_plm:
            self.feeze_paramter()  # knowledge prompt的参数在knowledge 模型初始化过程中完成
        

    def feeze_paramter(self):
        for name, param in self.pretrain_model.named_parameters():
            param.requires_grad = False
                

    def init_task_prompt(self):
        self.register_buffer('task_input_tokens', torch.arange(self.task_sequence_length).long())  # input_layer上面的prompt token

        self.task_wte_enc = nn.Embedding(self.task_sequence_length, self.n_embd)
        self.task_control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.task_mid_dim),
            nn.Tanh(),
            nn.Linear(self.task_mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        self.task_wte_dec = nn.Embedding(self.task_sequence_length, self.n_embd)
        self.task_control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.task_mid_dim),
            nn.Tanh(),
            nn.Linear(self.task_mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        self.task_wte_cross = nn.Embedding(self.task_sequence_length, self.n_embd)  # cross_attention的prompt
        self.task_control_trans_cross = nn.Sequential(
            nn.Linear(self.n_embd, self.task_mid_dim),
            nn.Tanh(),
            nn.Linear(self.task_mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        # 初始化
        if self.args.init_task == 'random':
            pass
        elif self.args.init_task == 'same':
            begin = self.args.gap
            end = self.args.gap + self.args.task_sequence_length                
            index = list(range(begin, end))
            init_prompt_value = self.pretrain_model.shared.weight[index].clone().detach()
            self.task_wte_enc.weight = nn.parameter.Parameter(init_prompt_value)
            self.task_wte_dec.weight = nn.parameter.Parameter(init_prompt_value)
            self.task_wte_cross.weight = nn.parameter.Parameter(init_prompt_value)   
        elif self.args.init_task == 'diff': 
            begin = self.args.gap
            end = self.args.gap + self.args.task_sequence_length                
            index = list(range(begin, end))
            init_prompt_value = self.pretrain_model.shared.weight[index].clone().detach()
            self.task_wte_enc.weight = nn.parameter.Parameter(init_prompt_value)
            begin = self.args.gap + self.args.task_sequence_length
            end = self.args.gap + self.args.task_sequence_length * 2                
            index = list(range(begin, end))
            init_prompt_value = self.pretrain_model.shared.weight[index].clone().detach()
            self.task_wte_dec.weight = nn.parameter.Parameter(init_prompt_value)
            begin = self.args.gap + self.args.task_sequence_length * 2
            end = self.args.gap + self.args.task_sequence_length * 3               
            index = list(range(begin, end))
            init_prompt_value = self.pretrain_model.shared.weight[index].clone().detach()
            self.task_wte_cross.weight = nn.parameter.Parameter(init_prompt_value)  

    def get_prompt(self, 
                   bsz=None, 
                   sample_size=1, 
                   cont_input_ids=None, 
                   cont_attention_mask=None,
                   domain=None,
                   np_seed=None,
                   not_generate=None,
                   ):
        old_bsz = bsz
        bsz = bsz * sample_size

        if self.args.use_task:

            # ====== encoder prompt
            # Encoder prefix
            input_tokens_enc = (
                self.task_input_tokens.unsqueeze(0).expand(old_bsz, -1)  # 不随着num_beams而变化
            )
            temp_control_enc = self.task_wte_enc(input_tokens_enc)

            past_key_values_enc = self.task_control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb

            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(
                bsz_enc,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)


            # ====== decoder prompt
            task_input_tokens = self.task_input_tokens.unsqueeze(0).expand(bsz, -1)
            temp_control_dec = self.task_wte_dec(task_input_tokens)

            past_key_values_dec = self.task_control_trans_dec(temp_control_dec)  # bsz, seqlen, layer*emb


            bsz, seqlen, _ = past_key_values_dec.shape
            past_key_values_dec = past_key_values_dec.view(
                bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
            )  
            # past_key_values [batch, prefix length, self.match_n_layer * 2, self.match_n_head, self.match_n_embd]
            past_key_values_dec = self.dropout(past_key_values_dec)
            past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)  # torch.Size([8, 10, 48, 16, 64])
            # tmp = past_key_values.permute([2, 0, 3, 1, 4])  # torch.Size([48, 8, 16, 10, 64])
            # tmp1 = tmp.split(2)  # 按Transformer层数维度，2个2个划分 按2划分的意义，一个是key，一个是value


            # ====== cross prompt
            temp_control_cross = self.task_wte_cross(task_input_tokens)

            past_key_values_cross = self.task_control_trans_cross(temp_control_cross)  # bsz, seqlen, layer*emb

            bsz, seqlen, _ = past_key_values_cross.shape
            past_key_values_cross = past_key_values_cross.view(
                bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
            )
            past_key_values_cross = self.dropout(past_key_values_cross)
            past_key_values_cross = past_key_values_cross.permute([2, 0, 3, 1, 4]).split(2)
            # 如果num_beams = 2 torch.Size([2, 16, 12, 5, 64])
        
        if self.args.use_knowledge:
            # 在一个 tuple 中前两个 tensor 为 self-attention 中的 key、value，
            # 后两个 tensor 为 cross-attention 中的 key、value
            past_key_values_knowledge_prompt, domain_loss = self.knowledge_prompt_encoder(
                cont_input_ids, 
                cont_attention_mask, 
                domain=domain, 
                np_seed=np_seed,
                not_generate=not_generate,
                )
            # 每个tensor torch.Size([8, 12, 10, 64])  10是token长度  torch.Size([8, 12, 512, 64])

        # 经过认定，num_beams=2时，只是复制了一遍

        # 这里必须要task
        if not self.args.use_task:
            return None
        result = []
        for i, key_val_enc in enumerate(past_key_values_enc):  # 之前对Transformer层数 按2划分

            # knowledge prompt
            if self.args.use_knowledge:
                key_val_know = past_key_values_knowledge_prompt[i]  # 如果是mlp映射，key_val_know是一个tensor
                knowledge_len = key_val_know[0].size()[2]

            temp = dict()  
            # encoder  torch.Size([2, 8, 12, 5, 64])
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous() if not self.args.use_knowledge else torch.concat((key_val_enc[0].contiguous(), key_val_know[0].contiguous()), dim=2),
                "prev_value": key_val_enc[1].contiguous() if not self.args.use_knowledge else torch.concat((key_val_enc[1].contiguous(), key_val_know[1].contiguous()), dim=2),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool() if not self.args.use_knowledge else torch.zeros(bsz_enc, (seqlen + knowledge_len)).to(key_val_enc.device).bool(),
            }

            # decoder
            key_val_dec = past_key_values_dec[i]
            temp["decoder_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),  # 按2划分的意义，一个是key，一个是value
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool()
                # bsz, preseqlen
            }

            # cross
            key_val_cross = past_key_values_cross[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_cross[0].contiguous(),
                "prev_value": key_val_cross[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_cross.device)
                    .bool(),
            }

            result.append(temp)

        return result, domain_loss


    def forward(self,
                ques_input_ids,
                ques_attention_mask,
                cont_input_ids=None,
                cont_attention_mask=None,
                labels=None,
                domain=None,
                np_seed=None,
                **kwargs,
                ):
        bsz = ques_input_ids.shape[0]


        past_prompt, domain_loss = self.get_prompt(
            bsz=bsz,
            cont_input_ids=cont_input_ids,
            cont_attention_mask=cont_attention_mask,
            domain=domain,
            np_seed=np_seed,
            not_generate=True,
        )

        output = self.pretrain_model(
            input_ids=ques_input_ids,
            attention_mask=ques_attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        )
        return {'loss': output.loss, 'domain_loss': domain_loss}

    def generate(self,
                ques_input_ids,
                ques_attention_mask,
                cont_input_ids=None,
                cont_attention_mask=None,
                domain=None,
                **kwargs):

        bsz = ques_input_ids.shape[0]


        past_prompt, domain_loss = self.get_prompt(
            bsz=bsz, 
            cont_input_ids=cont_input_ids,
            cont_attention_mask=cont_attention_mask,
            sample_size=kwargs['num_beams'],
            domain=domain,
            not_generate=False,
        )

        generated_ids = self.pretrain_model.generate(
            input_ids=ques_input_ids,
            attention_mask=ques_attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids