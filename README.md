# Rethinking Gradient Operator for Exposing AI-enabled Face Forgeries

Zhiqing Guo<sup>1</sup>, 
Gaobo Yang<sup>1</sup>,
Dengyong Zhang<sup>2</sup>,
and Ming Xia<sup>1</sup></br>
<sup>1</sup> Hunan University, China; 
<sup>2</sup> Changsha University of Science and Technology, China</br>

<img src="GocNet.png" alt="demo" width="800"/>

## GocNet
For image forensics, convolutional neural networks (CNNs) tend to learn image content features rather than subtle manipulation traces, which constrains detection performance. Existing works usually address this issue by following a common pipeline, namely subtracting the original pixel value from the predicted pixel value to enforce CNNs to learn more features from the manipulation traces. However, due to the complicated learning mechanism, they might still have some unnecessary performance losses. In this work, we rethink the advantages of image gradient operator in exposing AI-enabled face forgeries, and design two plug-and-play modules, namely tensor pre-processing (TP) and manipulation trace attention (MTA), by combining the gradient operator with CNNs. Specifically, the TP module refines the feature tensor of each channel in the network by the gradient operator to highlight manipulation traces and improve feature representation. Moreover, the MTA module considers two dimensions, namely channel and manipulation traces, to enforce the network to learn the distribution of the manipulation traces. Both modules can be seamlessly integrated into existing CNNs for end-to-end training. Experiments show that the proposed expert system achieves better results than prior works on five public image datasets.

## Requirements
- PyTorch 1.7.1
- Python 3.6.8

## Usage of GocNet.py
The GocNet needs to be placed in your training pipeline, you just need to use it like this:
```python
model = GocNet(num_classes)
```

## Citation
Please cite our paper if the code is used in your research:
```
@article{GUO2022119361,
title = {Rethinking gradient operator for exposing AI-enabled face forgeries},
journal = {Expert Systems with Applications},
pages = {119361},
year = {2022},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2022.119361},
url = {https://www.sciencedirect.com/science/article/pii/S095741742202379X},
author = {Zhiqing Guo and Gaobo Yang and Dengyong Zhang and Ming Xia},
}
```
