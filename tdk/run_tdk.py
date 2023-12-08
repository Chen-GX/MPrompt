import os, sys
import os.path as osp
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
os.chdir(sys.path[0])
sys.path.append("..")

from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from utils.arguments import get_args

from utils.train_utils import set_seed
from evaluation.evalution import evaluate
from model_class.tdk_model import T5tdkForConditionalGeneration
from utils.tdk_utils import save_tdk_prompt, get_parameter_number, load_tdkdata, get_group_parameters

logger = logging.getLogger(__name__)

# f1  exact_match  rouge_l
eval_dict = {
    'squad2': 'f1',
    'newsqa': 'rouge_l',
    'narrativeqa': 'rouge_l',
    'drop': 'f1',
    'mctest': 'exact_match',
    'boolq': 'exact_match',
    'boolq_np': 'exact_match',
    'arc_easy_with_ir': 'exact_match',
    'arc_hard_with_ir': 'exact_match',
    'openbookqa_with_ir': 'exact_match',
    'qasc_with_ir': 'exact_match',
    'race_middle': 'exact_match',
}

def train(args, epoch, tokenizer, model, device, loader, optimizer, scheduler):
    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    time1=time.time()
    train_total_loss = []
    train_loss = []
    train_domain_loss = []

    for _, data in enumerate(tqdm(loader)):

        ques_ids = data["ques_ids"].to(device, dtype=torch.long)
        ques_mask  = data["ques_mask"].to(device, dtype=torch.long)

        cont_ids = data["cont_ids"].to(device, dtype=torch.long)
        cont_mask  = data["cont_mask"].to(device, dtype=torch.long)

        labels = data["target_ids"].to(device, dtype=torch.long)  # 这里的y就有"" 259
        labels[labels == tokenizer.pad_token_id] = -100

        if args.use_domain:
            domain = data["domain"].to(device, dtype=torch.long)
        else:
            domain = None

        outputs = model.forward(
            ques_input_ids=ques_ids,
            ques_attention_mask=ques_mask,

            cont_input_ids=cont_ids,
            cont_attention_mask=cont_mask,
            # decoder_input_ids=y_ids,
            labels=labels,
            domain=domain,
            np_seed=_,
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        domain_loss = outputs["domain_loss"] if isinstance(outputs, dict) else outputs[1]

        if domain_loss != None:  # 可能会计算出0，从而跳过loss计算
            if domain_loss.item() == 0:
                domain_weight = args.domain_weight
            else:
                domain_weight = min(args.domain_weight, abs(loss.item() / domain_loss.item()))
                
            if args.domain_loss_name == 'cka':
                # 检查一下cka 是否需要动态学习率
                if args.cka_dynamic_weight:
                    total_loss = loss + domain_weight * domain_loss
                else:
                    total_loss = loss + args.domain_weight * domain_loss
            else:
                total_loss = loss - domain_weight * domain_loss
        else:
            total_loss = loss

        if _ % args.step_log == 0:
            time2=time.time()
            if domain_loss != None:
                # logger.info("epoch:"+str(epoch)+"-total_loss:"+str(total_loss.item())+"-loss:"+str(loss.item())+"-kl_loss:"+str(domain_loss.item())+";each step's time spent:"+str(float(time2-time1)/float(_+0.0001)))
                logger.info(f"epoch:{str(epoch)}-total_loss:{total_loss.item():.8f}-loss:{loss.item():.8f}-{args.domain_loss_name}_loss:{domain_loss.item():.8f};each step's time spent:{float(time2-time1)/float(_+0.0001):.8f}")
            else:
                logger.info(f"epoch:{str(epoch)}-total_loss:{total_loss.item():.8f};each step's time spent:{float(time2-time1)/float(_+0.0001):.8f}")
        if torch.isnan(total_loss).data:
            logger.info("Stop training because loss=%s" % (total_loss.data))
            return True
        
        train_total_loss.append(total_loss.item())
        if domain_loss != None:
            train_loss.append(loss.item())
            train_domain_loss.append(domain_loss.item())
        total_loss = total_loss / args.gradient_accumulation_steps 
        # optimizer.zero_grad()
        total_loss.backward()

        if ((_ + 1) % args.gradient_accumulation_steps == 0) or ((_ + 1 == len(loader))):
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()    # We have accumulated enought gradients
            if scheduler != None:
                scheduler.step()
                # lr_info = ''
                # for i in range(len(optimizer.param_groups)):
                #     lr_info += f"lr_{i} {optimizer.param_groups[i]['lr']} |"
                # logger.info(lr_info)
            optimizer.zero_grad()

    loss_info = f'epoch {epoch} avg train total loss {np.mean(train_total_loss):.8f}'
    if len(train_domain_loss) > 0:
        loss_info += f' | loss {np.mean(train_loss):.8f} | {args.domain_loss_name}_loss {np.mean(train_domain_loss):.8f}'
    logger.info(loss_info)
    return None

        
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
    set_seed(args.seed)
    device = torch.device('cuda')

    # load model
    model = T5tdkForConditionalGeneration(args)
    tokenizer = model.tokenizer

    # =====模型参数操作
    # TODO: 冻结参数
    model=model.to(device)
    param_group = get_group_parameters(args, model)
    # optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    get_parameter_number(model)

    logger.info("待训练的参数".center(50, "="))
    train_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_names.append(name)
            logger.info(f"{name}, {param.size()}")
    logger.info("".center(50, "="))

    # 加载数据
    train_loader, val_loader, test_loader = load_tdkdata(args, tokenizer)

    optimizer = torch.optim.AdamW(params=param_group)

    # warmup
    if args.warmup_ratio != 0.0:
        total_steps = len(train_loader) * args.epoch // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    else:
        scheduler = None

    best_eval = -1
    best_epoch = -1

    for epoch in range(args.epoch):
        # 1) train for one epoch
        train_nan = train(args, epoch, tokenizer, model, device, train_loader, optimizer, scheduler)
        if train_nan:
            break

        if args.do_valid:
            # 3) evaluating test dataset
            logger.info(f"[Initiating Validation]...")

            curr_eval = inference(args, tokenizer, model, device, val_loader, save_predictions=False)
            logger.info(f'epoch {epoch} | val eval: {curr_eval}')
            if best_eval < curr_eval[eval_dict[args.dataset_name]]:
                best_eval = curr_eval[eval_dict[args.dataset_name]]
                best_epoch = epoch
                # save model for best epoch
                logger.info(f"[Saving the best Model at {epoch}]...")
                path = os.path.join(args.output_dir, 'best_model')
                # save_model(model, tokenizer, path)
                save_tdk_prompt(model, path)


    metrics = {f'best_eval_{eval_dict[args.dataset_name]}': best_eval, 'best_epoch': best_epoch}
    # load the best model and inference
    if args.do_predict:

        logger.info(f"[load the best model at epoch_{best_epoch} and inference]...")
        path = os.path.join(args.output_dir, 'best_model')
        model = T5tdkForConditionalGeneration(args)
        model.load_state_dict(torch.load(osp.join(path, 'tdk_prompt.model')), strict=False)
        model = model.to(device) # model.knowledge_prompt_encoder.knowledge_t5.decoder_embed
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
    logger.info(f'train complete! {metrics}')

    with open(osp.join(args.output_dir, 'metrics.json'), "w", encoding='utf-8') as f: ## 设置'utf-8'编码
        f.write(json.dumps(metrics, ensure_ascii=False, indent=4))

    logger.info(f'output_path: {args.output_dir}')

            
    
if __name__ == '__main__':
    st=time.time()
    args = get_args()
    
    main(args)

    logger.info(f"done cost:{time.time()-st}")