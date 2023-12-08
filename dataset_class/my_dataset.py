import logging
import torch 
import pandas as pd
import os.path as osp
import os, json, string
import numpy as np
import re

logger = logging.getLogger(__name__)

class tdk_QA_dataset():
    def __init__(self, args, tokenizer, data_split='Train'):
        """
        """
        self.args = args
        self.file_path = osp.join(args.data_dir, args.dataset_name)
        self.tokenizer = tokenizer
        self.ques_len = args.max_ques_length
        self.cont_len = args.max_cont_length
        self.ans_len = args.max_ans_length
        self.data_split = data_split
        self.read_file(args)

    
    def read_file(self, args):
        # 从path中加载question，context，answer，domain label

        # 详细格式见readme
        data = pd.read_csv(osp.join(self.file_path, f'{self.data_split}_{self.args.domain_type}.tsv'), sep='\t', header=None, keep_default_na=False)
        self.ques_opt, self.context, self.answer, self.domain = data[0].to_list(), data[1].to_list(), data[2].to_list(), data[3].to_list()
        assert self.args.domain_size == len(list(set(self.domain)))
        if args.max_debug_samples > 0:
            self.ques_opt, self.context, self.answer, self.domain = self.ques_opt[:args.max_debug_samples], self.context[:args.max_debug_samples], self.answer[:args.max_debug_samples], self.domain[:args.max_debug_samples]
        logger.info(f"data_size: {len(self.ques_opt)}")
        
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.ques_opt)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        ques_opt = self.ques_opt[index]
        context = self.context[index]
        answer = self.answer[index]
        domain = self.domain[index]
        if self.args.ques_cont:
            ques_cont_input = ques_opt + " \\n " + context
        else:
            ques_cont_input =  ques_opt

        ques_input = self.tokenizer.batch_encode_plus(
            [ques_cont_input],
            max_length=self.ques_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        cont_input = self.tokenizer.batch_encode_plus(
            [context],
            max_length=self.cont_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ) 

        target = self.tokenizer.batch_encode_plus(
            [answer],
            max_length=self.ans_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        ques_ids = ques_input["input_ids"].squeeze()
        ques_mask = ques_input["attention_mask"].squeeze()

        cont_ids = cont_input["input_ids"].squeeze()
        cont_mask = cont_input["attention_mask"].squeeze()

        target_ids = target["input_ids"].squeeze()
        # target_mask = target["attention_mask"].squeeze()

        return {
            "ques_ids": ques_ids.to(dtype=torch.long),
            "ques_mask": ques_mask.to(dtype=torch.long),

            "cont_ids": cont_ids.to(dtype=torch.long),
            "cont_mask": cont_mask.to(dtype=torch.long),
            
            "target_ids": target_ids.to(dtype=torch.long),

            "domain": torch.tensor(domain, dtype=torch.long),

            "ques_opt": ques_opt,
            "context": context,
            "answer": answer,
        }





        