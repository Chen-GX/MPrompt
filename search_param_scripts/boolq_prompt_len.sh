#!/bin/bash  
MY_PYTHON="the path of python" # 这里是conda中具体环境下的python绝对路径名称
# 模型参数
model_name='unifiedqa-t5-base'
model_name_or_path='allenai/unifiedqa-t5-base'
# 数据集
data_dir="./qa_datasets"
dataset_name='boolq'
# 输出地址
output_dir="./search_output_dir/prompt_token"
max_debug_samples=0 #  不调试

EPOCH=10
batch_size=8
val_batch_size=8
warmup_ratio=0.1 # warm up
gradient_accumulation_steps=1  # 原文的batch size都特别小
step_log=$(( 30 * $gradient_accumulation_steps ))  # 原来的两倍 由于梯度累计
max_ques_length=512  # ques_opt 的长度
max_cont_length=512  # context 的长度
max_ans_length=10  # answer 的长度

# 结果生成相关参数
num_beams=2

# prompt 参数
use_task=True
use_domain=True
use_knowledge=True
ques_cont=True  # 输入是否包含cont
prompt_dropout=0.1
# task参数
task_sequence_length=10
lr=5e-5
init_task='random'
# knowledge 参数
knowledge_lr=$lr  # 这里改一下
# knowledge_sequence_length=15
map_hidden=True  # mlp映射hidden state,  False代表self past key values
kd_prompt_dropout=0.1
# domain参数
# n_prompt_tokens=5
domain_lr=$lr
gap=5
domain_size=3
domain_type=kmeans_context_3
loss_sample_n=3
domain_same_init=same  # prompt token是否同样的初始化  ['same', 'each_same', 'diff']  
use_enc_dec=False  # 一直是False
domain_weight=0.0001
domain_loss_name=cka  # kl mmd cka None 不进行约束
cka_dynamic_weight=False # 对cka是否采用动态学习率
gap_knowledge=False  # domain prompt初始化是否跳过knowledge  为True就跳过

EXEC=./tdk/run_tdk.py   # 这里配置实际运行的python文件名

export CUDA_VISIBLE_DEVICES="0"

# 需要变化的参数————prompt
prompt_lenght='5 10 15 20 30 40 50 60' #   

for knowledge_sequence_length in $prompt_lenght  #此处就不使用大括号了。
    do
        for n_prompt_tokens in $prompt_lenght
            do
                $MY_PYTHON $EXEC \
                --epoch $EPOCH \
                --batch_size $batch_size \
                --val_batch_size $val_batch_size \
                --lr $lr \
                --knowledge_lr $knowledge_lr \
                --data_dir $data_dir \
                --dataset_name $dataset_name \
                --step_log $step_log \
                --max_debug_samples $max_debug_samples \
                --output_dir $output_dir \
                --model_name $model_name \
                --model_name_or_path $model_name_or_path \
                --num_beams $num_beams \
                --gradient_accumulation_steps $gradient_accumulation_steps \
                --warmup_ratio $warmup_ratio \
                --use_task $use_task \
                --use_domain $use_domain \
                --use_knowledge $use_knowledge \
                --task_sequence_length $task_sequence_length \
                --knowledge_sequence_length $knowledge_sequence_length \
                --max_ques_length $max_ques_length \
                --max_cont_length $max_cont_length \
                --max_ans_length $max_ans_length \
                --ques_cont $ques_cont \
                --prompt_dropout $prompt_dropout \
                --domain_size $domain_size \
                --n_prompt_tokens $n_prompt_tokens \
                --domain_lr $domain_lr \
                --gap $gap \
                --domain_type $domain_type \
                --domain_same_init $domain_same_init \
                --loss_sample_n $loss_sample_n \
                --use_enc_dec $use_enc_dec \
                --domain_weight $domain_weight \
                --domain_loss_name $domain_loss_name \
                --cka_dynamic_weight $cka_dynamic_weight \
                --gap_knowledge $gap_knowledge \
                --kd_prompt_dropout $kd_prompt_dropout \
                --init_task $init_task \

        done
done