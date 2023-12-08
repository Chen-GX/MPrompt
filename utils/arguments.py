import os
import os.path as osp
import time
import argparse
from utils.log_utils import log_params

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def get_args():
    parser = argparse.ArgumentParser()  # 参数解释器
    # ============预训练模型参数
    parser.add_argument("--model_name", type=str, default="unifiedqa-t5-base")  
    parser.add_argument("--model_name_or_path", type=str, default="allenai/unifiedqa-t5-base") 
    parser.add_argument("--knowledge_model_name_or_path", type=str, default="allenai/unifiedqa-t5-small") 
    parser.add_argument("--cache_dir", type=str, default="./models")


    # ============训练参数
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=5)  # 每个GPU的batch_size数
    parser.add_argument("--val_batch_size", type=int, default=5)  # 每个GPU的batch_size数
    parser.add_argument("--max_ques_length", type=int, default=512)  # input 的最大长度
    parser.add_argument("--max_cont_length", type=int, default=512)  # input 的最大长度
    parser.add_argument("--max_ans_length", type=int, default=170)  # answer 的最大长度
    parser.add_argument("--max_debug_samples", type=int, default=10)  
    # optimazer 参数
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warm up ratio')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # 生成参数
    parser.add_argument('--num_beams', type=int, default=2)


    # ============数据集
    parser.add_argument("--data_dir", type=str, default="./qa_datasets")
    parser.add_argument("--dataset_name", type=str,
                         default='arc_easy_with_ir',
                        #  choices=['mctest', 'boolq', 'boolq_np', 'arc_easy_with_ir', 'arc_hard_with_ir'],
                         )

    # ===========文件参数
    parser.add_argument("--output_dir", type=str, default="./output_dir/test")
    parser.add_argument("--step_log", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1024)

    parser.add_argument("--do_valid", type=str2bool, default=True,
                    help='predict test_set in the end of each epoch')
    parser.add_argument("--do_predict", type=str2bool, default=True,
                    help='predict test_set in the end of each epoch')


    # ==========tdk参数
    parser.add_argument('--use_task', type=str2bool, default=True)  # task-specific prompt
    parser.add_argument('--use_domain', type=str2bool, default=True)  # domain-level prompt
    parser.add_argument('--use_knowledge', type=str2bool, default=True)  # knowledge prompt
    parser.add_argument('--ques_cont', type=str2bool, default=True)  # 输入时ques+cont
    parser.add_argument('--prompt_dropout', type=float, default=0.0)  # 对past key values prompt的dropout
    parser.add_argument('--freeze_plm', type=str2bool, default=True)  # 冻结预训练模型

    # ==========task-specific prompt参数
    parser.add_argument("--task_sequence_length", type=int, default=5)
    parser.add_argument("--task_mid_dim", type=int, default=512)  # prompt MLP中间层维度
    parser.add_argument("--init_task", type=str, default='random', choices=['random', 'same', 'diff'])
    
    # ==========knowledge prompt参数
    parser.add_argument("--knowledge_sequence_length", type=int, default=10)
    parser.add_argument("--knowledge_lr", type=float, default=1e-2)
    parser.add_argument("--map_hidden", type=str2bool, default=True,
                         help="Mapping via MLP using hidden output")
    parser.add_argument("--knowledge_mid_dim", type=int, default=512)  # prompt MLP中间层维度
    parser.add_argument('--kd_prompt_dropout', type=float, default=0.0) 

    # ==========domain prompt参数
    parser.add_argument("--domain_size", type=int, default=3)  # domain的规模
    parser.add_argument("--domain_lr", type=float, default=1e-2)
    parser.add_argument("--n_prompt_tokens", type=int, default=5)
    parser.add_argument("--use_encoder_prompt", type=str2bool, default=False)  # 一直是False
    parser.add_argument("--use_decoder_prompt", type=str2bool, default=True)
    parser.add_argument("--init_from_vocab", type=str2bool, default=True)  # prompt 的token数
    parser.add_argument("--gap", type=int, default=5)  # 初始化时跳过前面多少个单词 
    parser.add_argument("--domain_type", type=str, default="kmeans_context_3")  # train_kmeans_context_3
    parser.add_argument("--domain_same_init", type=str, default='same', choices=['same', 'each_same', 'diff'])
    parser.add_argument("--loss_sample_n", type=int, default=3)  # 采样的个数
    parser.add_argument("--use_enc_dec", type=str2bool, default=False)  # 一直是False
    parser.add_argument("--domain_weight", type=float, default=0.001)
    parser.add_argument("--domain_loss_name", type=str, default='kl', choices=['kl', 'mmd', 'cka', 'None'])
    parser.add_argument("--cka_dynamic_weight", type=str2bool, default=False)
    parser.add_argument("--gap_knowledge", type=str2bool, default=False)

    
    args = parser.parse_args()  # 解析参数
    # 时间戳后缀，
    timestamp = time.strftime("%m-%d_%H-%M-%S",time.localtime())
    args.model_type = 'tdk_'
    doc_name = ''
    if args.use_task:
        args.model_type += 't'
        doc_name += f'tlr_{args.lr}_'
    if args.use_domain:
        args.model_type += 'd'
        doc_name += f'dlr_{args.domain_lr}_ds_{args.domain_size}_'
    if args.use_knowledge:
        args.model_type += 'k'
        doc_name += f'klr_{args.knowledge_lr}_'
    
    
    doc_name += f'ep_{args.epoch}_bs_{args.batch_size}_wup_{args.warmup_ratio}_{timestamp}'
    
    args.output_dir = osp.join(args.output_dir, args.dataset_name, args.model_name, args.model_type, args.domain_loss_name, 'run', doc_name)


    log_params(args)
    
    return args