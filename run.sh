nohup python train.py \
--dataset imagenet \
--model ViT-B/32 \
--batch_size 256 \
--seed 0 \
--gpu_id "6" \
--augment_text \
--save_path "/path/to/save/" \
--lambda_sep 0.1 \
--lambda_con 0.01 \
--save_freq 500 \
--max_step 5000 \
--lr 5e-7 \
--eval \
> "/path/to/log/run.log" 2>&1 &