#!/bin/bash
#SBATCH --job-name=cl_nlp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --account=all
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/%u/CL/prog_prompt_%j.log

source ~/miniconda/bin/activate 
conda init
source activate nlp

HPARAMS=(
    "--task_list ag_news yelp_review_full amazon yahoo_answers_topics dbpedia --save_name prog_prompt_len10_1_order4 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    "--task_list ag_news yelp_review_full amazon yahoo_answers_topics dbpedia --save_name prog_prompt_len10_2_order4 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    "--task_list ag_news yelp_review_full amazon yahoo_answers_topics dbpedia --save_name prog_prompt_len10_3_order4 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"

    "--task_list ag_news yelp_review_full amazon yahoo_answers_topics dbpedia --save_name prog_prompt_len5_1_order4 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    "--task_list ag_news yelp_review_full amazon yahoo_answers_topics dbpedia --save_name prog_prompt_len5_2_order4 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    "--task_list ag_news yelp_review_full amazon yahoo_answers_topics dbpedia --save_name prog_prompt_len5_3_order4 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"


    "--task_list yelp_review_full yahoo_answers_topics amazon dbpedia ag_news --save_name prog_prompt_len10_1_order5 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    "--task_list yelp_review_full yahoo_answers_topics amazon dbpedia ag_news --save_name prog_prompt_len10_2_order5 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    "--task_list yelp_review_full yahoo_answers_topics amazon dbpedia ag_news --save_name prog_prompt_len10_3_order5 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"

    "--task_list yelp_review_full yahoo_answers_topics amazon dbpedia ag_news --save_name prog_prompt_len5_1_order5 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    "--task_list yelp_review_full yahoo_answers_topics amazon dbpedia ag_news --save_name prog_prompt_len5_2_order5 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    "--task_list yelp_review_full yahoo_answers_topics amazon dbpedia ag_news --save_name prog_prompt_len5_3_order5 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"


    "--task_list dbpedia yahoo_answers_topics ag_news amazon yelp_review_full --save_name prog_prompt_len10_1_order6 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    "--task_list dbpedia yahoo_answers_topics ag_news amazon yelp_review_full --save_name prog_prompt_len10_2_order6 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    "--task_list dbpedia yahoo_answers_topics ag_news amazon yelp_review_full --save_name prog_prompt_len10_3_order6 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"

    "--task_list dbpedia yahoo_answers_topics ag_news amazon yelp_review_full --save_name prog_prompt_len5_1_order6 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    "--task_list dbpedia yahoo_answers_topics ag_news amazon yelp_review_full --save_name prog_prompt_len5_2_order6 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    "--task_list dbpedia yahoo_answers_topics ag_news amazon yelp_review_full --save_name prog_prompt_len5_3_order6 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"


    # "--task_list yahoo_answers_topics ag_news yelp_review_full --save_name prog_prompt_len10_1_order3 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    # "--task_list yahoo_answers_topics ag_news yelp_review_full --save_name prog_prompt_len10_2_order3 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"
    # "--task_list yahoo_answers_topics ag_news yelp_review_full --save_name prog_prompt_len10_3_order3 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 10"

    # "--task_list yahoo_answers_topics ag_news yelp_review_full --save_name prog_prompt_len5_1_order3 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    # "--task_list yahoo_answers_topics ag_news yelp_review_full --save_name prog_prompt_len5_2_order3 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"
    # "--task_list yahoo_answers_topics ag_news yelp_review_full --save_name prog_prompt_len5_3_order3 --num_epochs 50 --freeze_weights 1 --freeze_except word_embeddings  --prompt_tuning 1 --prefix_len 5"


)

cmd="python train_cl2.py ${HPARAMS[SLURM_ARRAY_TASK_ID]} \
     --seq_len 450 --select_k_per_class 2000 --early_stopping 1 --one_head 0"

echo $cmd
eval $cmd
