---
license: apache-2.0
datasets:
- Chrisneverdie/SportsRWKV
language:
- en
pipeline_tag: text-generation
tags:
- sports
---
#### Use this model space for example inference
https://huggingface.co/spaces/Chrisneverdie/SportsRWKV

This model is built on RWKV 6 structure - an RNN with transformer-level LLM performance. It can be directly trained like a GPT (parallelizable). So it's combining the best of RNN and transformer - great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding.

This is also part of a project to explore the potential of the Small Language Model in the Sports domain.
Our previous projects:
  https://github.com/chrischenhub/FirstSportsELM
  https://huggingface.co/spaces/Chrisneverdie/SportsDPT
  
This model is finetuned with QA pairs so a text completion task may result in an error.
Questions unrelated to sports may suffer from poor performance.
It may still provide incorrect information so just take it as a toy domain model.

# SportsRWKV
  Created by Chris Zexin Chen
  
  Email for question: zc2404@nyu.edu


As avid sports enthusiasts, we’ve consistently observed a gap in the market for a dedicated
large language model tailored to the sports domain. This research stems from our intrigue
about the potential of a language model that is exclusively trained and fine-tuned on sports-
related data. We aim to assess its performance against generic language models, thus delving
into the unique nuances and demands of the sports industry

This model structure is built by BlinkDL: https://github.com/BlinkDL/RWKV-LM


### Pretrain Data 
https://huggingface.co/datasets/Chrisneverdie/SportsRWKV
*fixed_text_document.bin&fixed_text_document.idx ~8.4 Gb/4.5B tokens*


## Pretrain
For checkpoint file visit: https://huggingface.co/Chrisneverdie/SportsRWKV_150m

To replicate our model, you need to use fixed_text_document.bin & fixed_text_document.idx, which is processed and ready to train.
We trained on a 2xH100 80GB node for 5 hrs to get a val loss ~2.305. Once you set up the environment:

For best performance, use python 3.10, torch 2.1.2+cu121 (or latest), cuda 12.3+, latest deepspeed, but keep pytorch-lightning==1.9.5
best performance:
```
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
```
First create the model:
```
python train.py --wandb "" --proj_dir "output/"\
 --data_file "data/fixed_text_document" --data_type "binidx" --vocab_size 65536 --my_testing "x060"\
 --ctx_len 1024 --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 --my_exit_tokens 4534166811 --magic_prime 4427879 \
--lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0  --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0
```

Start training:
```
python train.py --load_model "0" --wandb "Test" --proj_dir "output/"
--my_testing "x060" --ctx_len 1024 --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 --data_file "data/fixed_text_document" --my_exit_tokens 4534166811 --magic_prime 4427879 \
--num_nodes 1 --micro_bsz 12 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 --lr_init 6e-4 --lr_final 6e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 \
--adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 --weight_decay 0.1 --epoch_save 5 --head_size_a 64 --accelerator gpu --devices 1 \
--precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 --enable_progress_bar True --ds_bucket_mb 200
```
Note: if you are using commercial GPUs, change --grad_cp to 1 and --ds_bucket_mb to 2. Change --devices/--num_nodes if you have multiple GPUs/nodes

The loss should look like this
![image/png](https://cdn-uploads.huggingface.co/production/uploads/656590bd40440ddcc051ade7/S3JLeK9A2fCxCz6W6qFib.png)

After you finish the training, the final .pth file will be saved under the output folder


## Fine Tune
We used thousands of GPT4-generated Sports QA pairs to finetune our model - specifics can be found under: https://github.com/chrischenhub/FirstSportsELM/tree/main/finetune

1. Convert TXT to Jsonl files
   
```python Json2Bin.py```

2. Convert Jsonl to Binidx for fine-tuning

```python make_data.py your_data.jsonl 3 1024```

3. Fine Tune the checkpoint with the following:
Note: put the pretrained .pth file under output
```
python train.py --load_model "0" --wandb "SportsRWKV_ft" --proj_dir "output/" --my_testing "x060" \
 --ctx_len 1024 --my_pile_stage 3 --epoch_count 360 --epoch_begin 0 \
 --data_file "data/test" --my_exit_tokens 1081350 --magic_prime 1049 \
 --num_nodes 1 --micro_bsz 16 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 \
 --lr_init 3e-3 --lr_final 3e-4 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.1 --epoch_save 10 --head_size_a 64 \
 --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 --enable_progress_bar True --ds_bucket_mb 2
```


## Inference
For inference, use: https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py


## Cost
The entire pretrain and finetune process costs around 100 USD. ~50$ in GPU rentals and ~50$ in OpenAI API usage.
