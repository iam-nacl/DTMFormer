# DTMFormer

This repo is the official implementation for:\
[AAAI2024] DTMFormer: Dynamic Token Merging for Boosting Transformer-Based Medical Image Segmentation.\
(The details of our DTMFormer can be found at the models directory in this repo or in the paper. We take SETR for example.)

## Requirements

* python 3.6
* pytorch 1.8.0
* torchvision 0.9.0

## Datasets

* The ACDC dataset could be acquired from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/). 
* The ISIC 2018 dataset could be acquired from [here](https://challenge.isic-archive.com/data/).
* The BTCV dataset could be acquired from [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789).  


## Training
Commands for training
```
python train.py
```
## Testing
Commands for testing
``` 
python test.py
```

## References

1. [vit-pytorch](https://github.com/lucidrains/vit-pytorch)