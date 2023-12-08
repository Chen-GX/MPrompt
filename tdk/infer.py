import os, sys
os.chdir(sys.path[0])
sys.path.append("..")
import os.path as osp
import argparse
import time
import json
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.log_utils import log_params
from evaluation.evalution import evaluate
from model_class.tdk_model import T5tdkForConditionalGeneration
from utils.tdk_utils import load_tdkdata

logger = logging.getLogger(__name__)

# f1  exact_match  rouge_l
eval_dict = {
    'squad2': 'f1',
    'newsqa': 'rouge_l',
    'narrativeqa': 'rouge_l',
    'drop': 'f1',
    'mctest': 'exact_match',
    'boolq': 'f1',
    'boolq_np': 'f1',
    'arc_easy_with_ir': 'exact_match',
    'arc_hard_with_ir': 'exact_match',
    'openbookqa_with_ir': 'exact_match',
    'qasc_with_ir': 'exact_match',
    'race_middle': 'exact_match',
}

def inference(args, tokenizer, model, device, loader, save_predictions=False):
    """
    Function to evaluate model for predictions
    """
    model.eval()
    predictions = []
    ques_opts = []
    contexts = []
    answers = []

    with torch.no_grad():
        for _, data in enumerate(tqdm(loader)):

            ques_ids = data["ques_ids"].to(device, dtype=torch.long)
            ques_mask  = data["ques_mask"].to(device, dtype=torch.long)

            cont_ids = data["cont_ids"].to(device, dtype=torch.long)
            cont_mask  = data["cont_mask"].to(device, dtype=torch.long)

            if args.use_domain:
                domain = data["domain"].to(device, dtype=torch.long)
            else:
                domain = None

            generated_ids = model.generate(
                ques_input_ids=ques_ids,
                ques_attention_mask=ques_mask,
                cont_input_ids=cont_ids,
                cont_attention_mask=cont_mask,
                domain=domain,
                num_beams=args.num_beams,
                min_length=1,
                max_length=args.max_ans_length,
                early_stopping=True,
                )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.extend(preds)
            answer = data['answer']
            answers.extend(answer)

            if save_predictions:
                ques_opt = data['ques_opt']  # 文字question + option
                context = data['context']
                ques_opts.extend(ques_opt)
                contexts.extend(context)
                

    if save_predictions:
        return predictions, ques_opts, contexts, answers
    
    return evaluate(args.dataset_name, predictions, answers)  # dict


def main(args):
    device = torch.device('cuda')
    metrics = {}
    # path = os.path.join(args.model_cache, 'best_model')
    model = T5tdkForConditionalGeneration(args)
    model.load_state_dict(torch.load(osp.join(args.model_cache, 'tdk_prompt.model')), strict=False)
    tokenizer = model.tokenizer
    model = model.to(device) # model.knowledge_prompt_encoder.knowledge_t5.decoder_embed
    # 加载数据
    train_loader, val_loader, test_loader = load_tdkdata(args, tokenizer)

    predictions, ques_opts, contexts, answers = inference(args, tokenizer, model, device, test_loader, save_predictions=True)
    test_eval = evaluate(args.dataset_name, predictions, answers)
    for k, v in test_eval.items():
        metrics[f'test_{k}'] = v
    logger.info(f"test {eval_dict[args.dataset_name]}: {metrics[f'test_{eval_dict[args.dataset_name]}']}")
    # 存储结果
    final_df = pd.DataFrame({"question": ques_opts, "context": contexts, "answer": answers, "predictions": predictions})
    path = args.output_dir
    os.makedirs(path, exist_ok=True)
    final_df.to_excel(os.path.join(path, f"predictions_test.xlsx"), index=None)
    final_df.to_csv(os.path.join(path, f"predictions_test.csv"), index=None)
    with open(osp.join(args.output_dir, 'metrics.json'), "w", encoding='utf-8') as f: ## 设置'utf-8'编码
        f.write(json.dumps(metrics, ensure_ascii=False, indent=4))



if __name__ == '__main__':
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v == 'True':
            return True
        if v == 'False':
            return False
    st=time.time()
    parser = argparse.ArgumentParser()
    # 测试模型的相关参数 allenai/unifiedqa-v2-t5-base-1363200 
    parser.add_argument("--model_name_or_path", type=str, default="allenai/unifiedqa-t5-base")  
    parser.add_argument("--knowledge_model_name_or_path", type=str, default="allenai/unifiedqa-t5-small") 
    parser.add_argument("--cache_dir", type=str, default="./models")
    
    parser.add_argument("--model_cache", type=str, default="./lambda_output/drop/unifiedqa-t5-base/tdk_tdk/cka/run/tlr_5e-05_dlr_5e-05_ds_3_klr_5e-05_ep_40_bs_16_lambda_1.0_06-02_08-32-08/best_model") 

    parser.add_argument("--batch_size", type=int, default=5)  # 每个GPU的batch_size数
    parser.add_argument("--val_batch_size", type=int, default=5)  # 每个GPU的batch_size数
    parser.add_argument("--data_dir", type=str, default="./qa_datasets")
    parser.add_argument("--dataset_name", type=str, default='drop')
    parser.add_argument("--max_ques_length", type=int, default=512)  # input 的最大长度
    parser.add_argument("--max_cont_length", type=int, default=512)  # input 的最大长度
    parser.add_argument("--max_ans_length", type=int, default=150)  # answer 的最大长度
    parser.add_argument("--max_debug_samples", type=int, default=0)  
    # 生成参数
    parser.add_argument('--num_beams', type=int, default=2)

    parser.add_argument("--output_dir", type=str, default="./lambda_output/drop/unifiedqa-t5-base/tdk_tdk/cka/run/tlr_5e-05_dlr_5e-05_ds_3_klr_5e-05_ep_40_bs_16_lambda_1.0_06-02_08-32-08")

        # ==========tdk参数
    parser.add_argument('--use_task', type=str2bool, default=True)  # task-specific prompt
    parser.add_argument('--use_domain', type=str2bool, default=True)  # domain-level prompt
    parser.add_argument('--use_knowledge', type=str2bool, default=True)  # knowledge prompt
    parser.add_argument('--ques_cont', type=str2bool, default=True)  # 输入时ques+cont
    parser.add_argument('--prompt_dropout', type=float, default=0.0)  # 对past key values prompt的dropout
    parser.add_argument('--freeze_plm', type=str2bool, default=True)  # 冻结预训练模型

    # ==========task-specific prompt参数
    parser.add_argument("--task_sequence_length", type=int, default=20)
    parser.add_argument("--task_mid_dim", type=int, default=512)  # prompt MLP中间层维度
    parser.add_argument("--init_task", type=str, default='random', choices=['random', 'same', 'diff'])
    
    # ==========knowledge prompt参数
    parser.add_argument("--knowledge_sequence_length", type=int, default=60)
    # parser.add_argument("--knowledge_lr", type=float, default=1e-2)
    parser.add_argument("--map_hidden", type=str2bool, default=True,
                         help="Mapping via MLP using hidden output")
    parser.add_argument("--knowledge_mid_dim", type=int, default=512)  # prompt MLP中间层维度
    parser.add_argument('--kd_prompt_dropout', type=float, default=0.0) 

    # ==========domain prompt参数
    parser.add_argument("--domain_size", type=int, default=3)  # domain的规模
    # parser.add_argument("--domain_lr", type=float, default=1e-2)
    parser.add_argument("--n_prompt_tokens", type=int, default=30)
    parser.add_argument("--use_encoder_prompt", type=str2bool, default=False)  # 一直是False
    parser.add_argument("--use_decoder_prompt", type=str2bool, default=True)
    parser.add_argument("--init_from_vocab", type=str2bool, default=True)  # prompt 的token数
    parser.add_argument("--gap", type=int, default=5)  # 初始化时跳过前面多少个单词 
    parser.add_argument("--domain_type", type=str, default="kmeans_context_3")  # train_kmeans_context_3
    parser.add_argument("--domain_same_init", type=str, default='same', choices=['same', 'each_same', 'diff'])
    parser.add_argument("--loss_sample_n", type=int, default=3)  # 采样的个数
    parser.add_argument("--use_enc_dec", type=str2bool, default=False)  # 一直是False
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--domain_loss_name", type=str, default='cka', choices=['kl', 'mmd', 'cka', 'None'])
    parser.add_argument("--cka_dynamic_weight", type=str2bool, default=False)
    parser.add_argument("--gap_knowledge", type=str2bool, default=False)

    args = parser.parse_args()
    # 时间戳后缀，
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
    args.output_dir = osp.join(args.output_dir, 'inference', timestamp)
    log_params(args)
    main(args)

    print(f"done cost:{time.time()-st}")