python train.py \
    --gpus=1 \
    --wandb \
    --entity=syip0005 \
    --epochs=1000 \
    --dataset=CERN \
    --model=SCSAE \
    --batch=1024 \
    --learning_rate_style=constant \
    --lr=1e-3 \
    --latent_n=48 \
    --batchnorm

python train.py \
--gpus=1 \
--wandb \
--entity=syip0005 \
--epochs=1000 \
--dataset=CERN \
--model=SCSAE \
--batch=1024 \
--learning_rate_style=constant \
--lr=1e-3 \
--latent_n=36 \
--batchnorm

python train.py \
    --gpus=1 \
    --wandb \
    --entity=syip0005 \
    --epochs=1000 \
    --dataset=CERN \
    --model=SCSAE \
    --batch=1024 \
    --learning_rate_style=constant \
    --lr=1e-4 \
    --latent_n=24 \
    --batchnorm

python train.py \
--gpus=1 \
--wandb \
--entity=syip0005 \
--epochs=1000 \
--dataset=CERN \
--model=SCSAE \
--batch=1024 \
--learning_rate_style=constant \
--lr=1e-3 \
--latent_n=144 \
--batchnorm