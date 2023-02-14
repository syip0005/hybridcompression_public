# Compression of electrical energy data, a hybrid deep learning approach

This is the working git for my Master's thesis research project at Monash University (Australia).

The project is to explore and evaluate the effectiveness (in terms of compression ratio, speed and reconstruction error) of deep learning architectures which have not yet beeen applied for compression.

The final research paper is available [here](https://drive.google.com/file/d/1pEdahC0OKNK6N3HnATcUX5YqJu3DCz1g/view)

The architectures include:
* Transformer-based models; and
* RNN-based model; and
* RNN and CNN compression models.

## Evaluation Criteria

* Volume Compressed: Compression Ratio, or Space Saving
* Reconstruction Error: MAE, or RSME

## Getting Started

Begin by downloading the datasets required using the shell script provided:

```sh
chmod +x ./dataset/download_dataset.sh
./dataset/download_dataset.sh
```

## Training

Training and testing is done via the `train.py` file. This file has been configured for CLI usage.

A sample execution is shown below.

```
python train.py --gpus=1 --epochs=1000 --dataset=CERN3 --model=HAE2 --batch=512 --learning_rate_style=constant \
--lr=1e-3 --latent_n=576 --batchnorm --weight_decay=1e-10 --dropout=0.1 --inputdays=12 --wandb --entity=syip0005 --optimiser=Adam
```

The full list of arguments are:

* gpus: number of GPUs on a single node.
* wandb: whether to use Weights and Biases (https://wandb.ai/site) to track results. If not included, then will NOT use WANDB.
* entity: WANDB login details.
* dataset: choose the dataset between `[CERN, CERN2, CERN3, DRED, UMASS, UMASS]` - details of differences are in `./lib/utils/dataloader.py`. Generally it is advantageous to use the latest versions.
* model: choose the model to use between `['AE', 'SCSAE', 'HAE', 'HAE2', 'HAE3']`, i.e., (base autoencoder, convoultional autoencoder and my hybrid autoencoder (also called CRAE in the paper) - details of differences are in `./lib/models/model.py`
* batch: training batch size (heuristically base 2)
* epochs: training epochs to run.
* lr: learning rate.
* weight_decay: weight decay.
* loss: only MSE implemented (as this is regression prediction effectively), so use MSE - but can also experiment with other losses.
* learning_rate_style: choose between `['constant', 'OneCycleLR']`, for OneCycleLR info https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html.
* optimiser: only `Adam` implemented.
* latent_n: dimension of latent vector for embedded representation (i.e., the compression)
* dropout: dropout probability for learning.
* inputdays: number of input days (for AE, this is length * daily_freq, for HAE/CAE this is rows of dataframe).
* batchnorm: whether to batchnorm or not. If not included, then will NOT batchnorm.
* noise: how to augment input with noise for denoising AE, choose between `['none', 'gauss', 'speckle']`.
* noise_pct: how much noise.
* dred_freq: the wide frequency for DRED dataset (as discussed in the paper).


## Authors

* **Scott Yip**

## Acknowledgements

* Dr. Hao Wang
