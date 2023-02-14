# Compression of electrical energy data, a hybrid deep learning approach

This is the working git for my Master's thesis research project at Monash University (Australia).

The project is to explore and evaluate the effectiveness (in terms of compression ratio, speed and reconstruction error) of deep learning architectures which have not yet beeen applied for compression.

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

## Authors

* **Scott Yip**

## Acknowledgements

* Dr. Hao Wang
