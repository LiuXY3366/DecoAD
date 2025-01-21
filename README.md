# DecoAD

![Framework_Overview](./data/pipeline.png)

## Getting Started

This code was tested on `Ubuntu 22.04.4 LTS` and requires:
* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### Setup Conda Environment:
```
git clone https://github.com/LiuXY3366/DecoAD
cd DecoAD

# Conda environment setup
conda env create -f environment.yml
conda activate DecoAD
```

### Data Directory
Data folder, including extracted poses and GT, can refer to [link](https://github.com/orhir/STG-NF/) for the data format. 

## Training/Testing
Training and Evaluating is run using:
```
python UNVAD/main_unsupervised.py    # Unsuperised

python WSVAD/main_unsupervised.py    # Weakly-/Funlly-superised
```

Evaluation of our pretrained model can be done using:
```
python UNVAD/stage1/test.py    # Unsuperised

python WSVAD/stage1/test.py    # Weakly-/Funlly-superised
```


## UFSR Dataset
To the best of our knowledge, this is the first dataset featuring dynamic scenes and incorporating scene-related anomalies.

![Demo](./data/demo.png)

The dataset has been placed at [link](https://pan.baidu.com/s/1gFGQmdg_AjEoZIf2yPGzPQ?pwd=6mx6).
