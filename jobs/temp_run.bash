python train.py --gpus=1 --epochs=1000 --dataset=DRED --model=HAE3 --batch=512 --learning_rate_style=constant \
--lr=1e-3 --latent_n=900 --batchnorm --weight_decay=0 --inputdays=60 --dred_freq=60