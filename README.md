# False-Claims-Against-Model-Ownership-Resolution
Pytorch implement of the paper False Claims against Model Ownership Resolution which is to appear in USENIX Sec '24.
[https://arxiv.org/pdf/2304.06607.pdf](https://arxiv.org/pdf/2304.06607.pdf)


## Dependencies
For [Adi](https://arxiv.org/abs/1802.04633) and [EWE](https://arxiv.org/abs/2002.12200), we use the [Watermark-Robustness-ToolBox](https://github.com/dnn-security/Watermark-Robustness-Toolbox) to reproduce their results; for [Li (b)](https://github.com/zhenglisec/Blind-Watermark-for-DNN), [DAWN](https://github.com/ssg-research/dawn-dynamic-adversarial-watermarking-of-neural-networks) , and [DI](https://github.com/cleverhans-lab/dataset-inference), we use their open-sourced implementations; as the source code of [Lukas](https://github.com/ayberkuckun/DNN-Fingerprinting) is based on TensorFlow (whereas all others are on Pytorch). We re-implemented their schemes in Pytorch.

To install dependencies
```
pip install -r requirements.txt
```

And we use the [mlconfig](https://github.com/narumiruna/mlconfig) to pass configuration parameters to each script. The configuration files used in our paper can be found in `configs/`.

The models for CIFAR10 and ImageNet can be downloaded from [this link](https://drive.google.com/drive/folders/1h1NcupuTF76XOdOY-CZWR_XMe4GJdmWx?usp=sharing). Save the corresponding models in `checkpoint/` like `checkpoint/cifar10`. The ten classes dataset of ImageNet we use are reported in our [research report](https://arxiv.org/pdf/2304.06607.pdf). It can also be downloaded from the link above. Save the `small_imagenet` file to `~/data/`.


### Claim Generation and Verification
#### To train your own models

```
python train.py -c ['dir to the configure file']
```
For CIFAR-10
```
python train.py -c configs/cifar10/train/resnet.yaml
```
For ImageNet
```
python train.py -c configs/imagenet/train/resnet.yaml
```

#### To generate our false claims and test the MOR accuracy:


```
python attack.py --c ['dir to the configure file'] 
```
For CIFAR-10
```
python attack.py --c configs/cifar10/train/resnet.yaml
```
For ImageNet
```
python attack.py --c configs/imagenet/train/resnet.yaml
```

#### To generat false claims with different structures and the same data

For CIFAR-10
```
python attack.py --c configs/cifar10/train/resnet.yaml --adv False
```
For ImageNet
```
python attack.py --c configs/imagenet/train/resnet.yaml --adv False
```

#### To generate false claims with the same structure and different data
For CIFAR-10
```
python attack.py --c configs/cifar10/train/resnet_same_struct.yaml
```
For ImageNet
```
python attack.py --c configs/imagenet/train/resnet_same_struct.yaml
```

#### To generate the targeted false claims, add `adv_target True', for example, to generate targeted false claims with the same structure and different data for CIFAR-10
```
python attack.py --c configs/cifar10/train/resnet_same_struct.yaml --adv_target True
```


To compute the decision thresholds, 
for [Li (b)](https://github.com/zhenglisec/Blind-Watermark-for-DNN), [DAWN](https://github.com/ssg-research/dawn-dynamic-adversarial-watermarking-of-neural-networks), [Lukas](https://github.com/ayberkuckun/DNN-Fingerprinting) and  [DI](https://github.com/cleverhans-lab/dataset-inference), we recommend their official implements; and [Watermark-Robustness-ToolBox](https://github.com/dnn-security/Watermark-Robustness-Toolbox) for [Adi](https://arxiv.org/abs/1802.04633) and [EWE](https://arxiv.org/abs/2002.12200).

We also have a simple implementation respectively for Li(b), DAWN, Lukas and DI in the `defence/` to compute the decision thresholds.

Note that for Lukas, we first train surrogate models.

```
python train_surrogate.py -c configs/cifar10/train/resnet_lukas.yaml
```