import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig
from .knowledge_t5model import T5Model
from .loss_class import mmd_rbf, CudaCKA

class knowledge_prompt(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        config = self.get_domain_knowledge_config(args)
        self.config = config
        self.knowledge_sequence_length = args.knowledge_sequence_length
        self.register_buffer('knowledge_input_tokens', torch.arange(self.knowledge_sequence_length).long())
        # self.pad_embed = torch.nn.Embedding(1, config.d_model)
        self.knowledge_embed = torch.nn.Embedding(self.knowledge_sequence_length, config.d_model)

        self.knowledge_pretrain_model = T5Model.from_pretrained(
            args.knowledge_model_name_or_path, 
            cache_dir=args.cache_dir,
            config=config,
            local_files_only=True,
            )
        
        QA_config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, local_files_only=True)
        
        if args.map_hidden and isinstance(self.knowledge_pretrain_model, (T5Model)):
            # 通过decoder的output hidden映射prompt
            self.match_n_layer = QA_config.num_decoder_layers
            self.match_n_head = QA_config.num_heads
            self.n_embd = config.d_model  # 模型embd长度
            self.match_n_embd = QA_config.d_kv
            
            # output.last_hidden_state  # torch.Size([8, 10, 768])
            self.knowledge_control_trans = nn.Sequential(
                nn.Linear(self.n_embd, args.knowledge_mid_dim),
                nn.Tanh(),
                nn.Linear(args.knowledge_mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        
        # self.knowledge_t5.decoder.decoder_embed.weight
        # 初始化knowledge prompt参数
        with torch.no_grad():
            init_prompt_value = self.knowledge_pretrain_model.shared.weight[5:self.knowledge_sequence_length+5].clone().detach()
            self.knowledge_embed.weight = nn.parameter.Parameter(init_prompt_value)
        # 初始化domain prompt 参数
        if config.use_domain:
            self.initialize_domain_prompt()

        self.dropout = nn.Dropout(args.kd_prompt_dropout)
        # 冻结其余参数
        self.freeze_model_params()


    def initialize_domain_prompt(self):
        if self.args.init_from_vocab:
            if self.args.domain_same_init == 'same':
                if self.args.gap_knowledge:
                    begin = self.args.gap + self.knowledge_sequence_length
                    end = self.args.gap + self.knowledge_sequence_length + self.args.n_prompt_tokens
                else:
                    begin = self.args.gap
                    end = self.args.gap + self.args.n_prompt_tokens                    
                index = list(range(begin, end))
                init_prompt_value = self.knowledge_pretrain_model.shared.weight[index].clone().detach()
                for i in range(self.args.domain_size):
                    if self.args.use_encoder_prompt:
                        # Initialize weight
                        self.knowledge_pretrain_model.encoder.prompt_encoder_list[i].weight = nn.parameter.Parameter(init_prompt_value)
                    if self.args.use_decoder_prompt:
                        # Initialize weight
                        self.knowledge_pretrain_model.decoder.prompt_decoder_list[i].weight = nn.parameter.Parameter(init_prompt_value)      
            else:
                NotImplementedError("{} not implement".format(self.args.domain_same_init))

    def get_domain_knowledge_config(self, args):
        config = AutoConfig.from_pretrained(args.knowledge_model_name_or_path, cache_dir=args.cache_dir, local_files_only=True)
        config.update({
            'domain_size': args.domain_size,
            'use_domain': args.use_domain,
            'use_enc_dec': args.use_enc_dec,  # False
            'domain_loss_name': args.domain_loss_name,  # 损失名
            'loss_sample_n': args.loss_sample_n,
            'use_encoder_prompt': True  if args.use_domain and args.use_encoder_prompt else False,
            'use_decoder_prompt': True  if args.use_domain and args.use_decoder_prompt else False,
            'init_from_vocab': args.init_from_vocab,
            'pre_seq_len': args.n_prompt_tokens,
        })
        return config
    

    def freeze_model_params(self):
        for name, param in self.knowledge_pretrain_model.named_parameters():
            if "prompt_decoder_list" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def forward(self, cont_input_ids, cont_attention_mask, domain=None, np_seed=None, not_generate=True):
        device = cont_input_ids.device
        batch_size, cont_maxlen = cont_attention_mask.size()

        decoder_input_ids = self.knowledge_input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        # decoder_input_ids = self.knowledge_pretrain_model._shift_right(decoder_input_ids)
        decoder_input_embed = self.knowledge_embed(decoder_input_ids)

        output = self.knowledge_pretrain_model(
            input_ids=cont_input_ids, 
            attention_mask=cont_attention_mask, 
            decoder_inputs_embeds=decoder_input_embed,
            domain=domain,
            )  # 需要domain 和np_seed
        # output 的loss为None，因为并没有labels输入
        # 在一个 tuple 中前两个 tensor 为 self-attention 中的 key、value，
        # 后两个 tensor 为 cross-attention 中的 key、value


        domain_loss = None
        if self.config.use_domain and not_generate:
            assert self.config.loss_sample_n <= self.config.domain_size, 'error sample size > domain size'
            if self.config.loss_sample_n >= 1:
                # 如果只采样1个就不用对比了
                if self.config.domain_loss_name == 'mmd':
                    assert False
                    domain_loss = self.mmd_loss(np_seed)
                elif self.config.domain_loss_name == 'kl':
                    domain_loss = self.kl_loss(np_seed)
                elif self.config.domain_loss_name == 'cka':
                    domain_loss = self.cka_loss(np_seed)
                elif self.config.domain_loss_name == 'None':
                    # 如果没有就为None, 即不进行约束，可以实现使用use domain，但不约束的方式
                    pass
                else:
                    NotImplementedError('domain loss {} not implement'.format(self.config.domain_loss_name))


        if self.args.map_hidden:
            # 传入到encoder
            past_key_values_knowledge = self.knowledge_control_trans(output.last_hidden_state)  # bsz, seqlen, layer*emb

            bsz, seqlen, _ = past_key_values_knowledge.shape
            past_key_values_knowledge = past_key_values_knowledge.view(
                bsz,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values_knowledge = self.dropout(past_key_values_knowledge)
            past_key_values_knowledge = past_key_values_knowledge.permute([2, 0, 3, 1, 4]).split(2)


            return (past_key_values_knowledge, domain_loss)
        else:
            return (output.past_key_values, domain_loss)
    
    def cka_loss(self, np_seed):
        rng = np.random.default_rng(np_seed)
        prompt_tokens = self.knowledge_pretrain_model.decoder.prompt_tokens
        x_idx = rng.choice(np.arange(self.config.domain_size), size=self.config.loss_sample_n, replace=False)
        x_idx = list(x_idx)
        y_idx = x_idx[1:] + [x_idx[0]]

        cka = CudaCKA(device=self.knowledge_pretrain_model.device)
        all_cka_loss = 0
        loss_num = 0
        if self.config.domain_size > 1:  # 只有一个domain，不用拉开内部距离
            if self.config.use_encoder_prompt:
                assert False
                # encoder内部的
                enc_cka = 0
                for i, j in zip(x_idx, y_idx):
                    embed_i = self.knowledge_pretrain_model.encoder.prompt_encoder_list[i](prompt_tokens)
                    embed_j = self.knowledge_pretrain_model.encoder.prompt_encoder_list[j](prompt_tokens)
                    enc_cka += cka.kernel_CKA(embed_i, embed_j, sigma=None)
                enc_cka /= len(x_idx)
                all_cka_loss += enc_cka
                loss_num += 1

            # decoder内部的，decoder一定有
            dec_cka = 0
            for i, j in zip(x_idx, y_idx):
                embed_i = self.knowledge_pretrain_model.decoder.prompt_decoder_list[i](prompt_tokens)
                embed_j = self.knowledge_pretrain_model.decoder.prompt_decoder_list[j](prompt_tokens)
                dec_cka += cka.kernel_CKA(embed_i, embed_j, sigma=None)
            dec_cka /= len(x_idx)

            all_cka_loss = dec_cka
            loss_num += 1

        if self.config.use_enc_dec:
            assert False
            # enc_dec 之间的
            enc_dec_mmd = 0
            for i in x_idx:  # 对enc，dec之间相同的prompt进行约束
                embed_enc = self.encoder.prompt_encoder_list[i](prompt_tokens)
                embed_dec = self.decoder.prompt_decoder_list[i](prompt_tokens)
                enc_dec_mmd += cka.kernel_CKA(embed_enc, embed_dec, sigma=None)
            enc_dec_mmd /= len(x_idx)
            all_cka_loss += enc_dec_mmd
            loss_num += 1

        all_cka_loss /= loss_num
        return all_cka_loss


    def mmd_loss(self, np_seed):
        rng = np.random.default_rng(np_seed)
        prompt_tokens = self.encoder.prompt_tokens
        x_idx = rng.choice(np.arange(self.config.domain_size), size=self.config.loss_sample_n, replace=False)
        x_idx = list(x_idx)
        y_idx = x_idx[1:] + [x_idx[0]]
        all_mmd_loss = 0
        loss_num = 0
        if self.config.domain_size > 1:  # 只有一个domain，不用拉开内部距离
            # encoder内部的
            enc_mmd = 0
            for i, j in zip(x_idx, y_idx):
                embed_i = self.encoder.prompt_encoder_list[i](prompt_tokens)
                embed_j = self.encoder.prompt_encoder_list[j](prompt_tokens)
                enc_mmd += mmd_rbf(embed_i, embed_j)
            enc_mmd /= len(x_idx)
            # decoder内部的
            dec_mmd = 0
            for i, j in zip(x_idx, y_idx):
                embed_i = self.decoder.prompt_decoder_list[i](prompt_tokens)
                embed_j = self.decoder.prompt_decoder_list[j](prompt_tokens)
                dec_mmd += mmd_rbf(embed_i, embed_j)
            dec_mmd /= len(x_idx)

            all_mmd_loss = enc_mmd + dec_mmd
            loss_num += 2

        if self.config.use_enc_dec:
            # enc_dec 之间的
            enc_dec_mmd = 0
            for i in x_idx: # 对enc，dec之间相同的prompt进行约束
                embed_enc = self.encoder.prompt_encoder_list[i](prompt_tokens)
                embed_dec = self.decoder.prompt_decoder_list[i](prompt_tokens)
                enc_dec_mmd += mmd_rbf(embed_enc, embed_dec)
            enc_dec_mmd /= len(x_idx)
            all_mmd_loss += enc_dec_mmd
            loss_num += 1

        all_mmd_loss /= loss_num
        return all_mmd_loss

    def kl_loss(self, np_seed):  # array([0, 2, 1])  # array([2, 1, 0])
        rng = np.random.default_rng(np_seed)
        prompt_tokens = self.knowledge_pretrain_model.decoder.prompt_tokens.unsqueeze(0).to(self.knowledge_pretrain_model.device)
        x_idx = rng.choice(np.arange(self.config.domain_size), size=self.config.loss_sample_n, replace=False)
        x_idx = list(x_idx)
        y_idx = x_idx[1:] + [x_idx[0]]
        all_kl_loss = 0
        loss_num = 0
        if self.config.domain_size > 1:  # 只有一个domain，不用拉开内部距离
            if self.config.use_encoder_prompt:
                # encoder内部的
                enc_kl = 0
                for i, j in zip(x_idx, y_idx):
                    embed_i = self.knowledge_pretrain_model.encoder.prompt_encoder_list[i](prompt_tokens)
                    embed_j = self.knowledge_pretrain_model.encoder.prompt_encoder_list[j](prompt_tokens)
                    enc_kl += F.kl_div(F.log_softmax(embed_i, dim=-1), F.softmax(embed_j, dim=-1), reduction='mean')  # 每个位置的差异
                    enc_kl += F.kl_div(F.log_softmax(embed_j, dim=-1), F.softmax(embed_i, dim=-1), reduction='mean')
                enc_kl /= (len(x_idx) * 2)
                all_kl_loss += enc_kl
                loss_num += 1

            # decoder 内部的
            dec_kl = 0
            for i, j in zip(x_idx, y_idx):
                embed_i = self.knowledge_pretrain_model.decoder.prompt_decoder_list[i](prompt_tokens)
                embed_j = self.knowledge_pretrain_model.decoder.prompt_decoder_list[j](prompt_tokens)
                dec_kl += F.kl_div(F.log_softmax(embed_i, dim=-1), F.softmax(embed_j, dim=-1), reduction='mean')
                dec_kl += F.kl_div(F.log_softmax(embed_j, dim=-1), F.softmax(embed_i, dim=-1), reduction='mean')
            dec_kl /= (len(x_idx) * 2)

            all_kl_loss += dec_kl
            loss_num += 1

        # domain为1时，enc_dec之间的可以正常计算
        if self.config.use_enc_dec:
            assert False
            # enc_dec 之间的
            enc_dec_kl = 0
            for i in x_idx:  # 对enc，dec之间相同的prompt进行约束
                embed_enc = self.encoder.prompt_encoder_list[i](prompt_tokens)
                embed_dec = self.decoder.prompt_decoder_list[i](prompt_tokens)
                enc_dec_kl += F.kl_div(F.log_softmax(embed_enc, dim=-1), F.softmax(embed_dec, dim=-1), reduction='mean') 
                enc_dec_kl += F.kl_div(F.log_softmax(embed_dec, dim=-1), F.softmax(embed_enc, dim=-1), reduction='mean')
            enc_dec_kl /= (len(x_idx) * 2)
            all_kl_loss += enc_dec_kl
            loss_num += 1
        
        all_kl_loss /= loss_num
        return all_kl_loss