#!/bin/bash
#SBATCH --job-name=prompt_tune
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --account=all
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/%u/T5_prompts/single_task_lester_%j.log

source ~/miniconda/bin/activate 
conda init
source activate nlp

HPARAMS=(
    "--task_list cb --prefix_len 10 --save_name cb_k1000_10epochs_prog1 --select_k_per_class 1000 --num_epochs 10"
    "--task_list cb --prefix_len 10 --save_name cb_k1000_10epochs_prog2 --select_k_per_class 1000 --num_epochs 10"
    "--task_list cb --prefix_len 10 --save_name cb_k1000_10epochs_prog3 --select_k_per_class 1000 --num_epochs 10"

    # "--task_list cb --prefix_len 10 --save_name cb_k1000_10epochs_prog1 --select_k_per_class 1000 --num_epochs 10"
    # "--task_list cb --prefix_len 10 --save_name cb_k1000_10epochs_prog2 --select_k_per_class 1000 --num_epochs 10"
    # "--task_list cb --prefix_len 10 --save_name cb_k1000_10epochs_prog3 --select_k_per_class 1000 --num_epochs 10"

    # "--task_list cb --prefix_len 10 --save_name cb_k200_30epochs_prog1 --select_k_per_class 200 --num_epochs 30"
    # "--task_list cb --prefix_len 10 --save_name cb_k200_30epochs_prog2 --select_k_per_class 200 --num_epochs 30"
    # "--task_list cb --prefix_len 10 --save_name cb_k200_30epochs_prog3 --select_k_per_class 200 --num_epochs 30"

    # "--task_list cb --prefix_len 10 --save_name cb_k20_100epochs_prog1 --select_k_per_class 20 --num_epochs 100"
    # "--task_list cb --prefix_len 10 --save_name cb_k20_100epochs_prog2 --select_k_per_class 20 --num_epochs 100"
    # "--task_list cb --prefix_len 10 --save_name cb_k20_100epochs_prog3 --select_k_per_class 20 --num_epochs 100"
)

cmd="python train_t5_cl.py ${HPARAMS[SLURM_ARRAY_TASK_ID]} \
    --lr 0.3 --freeze_weights 1 --freeze_except xxxxxx --model_name t5-large --early_stopping 1 \
    --save_dir /data/home/arazdai/T5_prompts/T5_continual/T5_large_final/per_task_prompts/"

echo $cmd
eval $cmd
# cb mnli qnli multirc
