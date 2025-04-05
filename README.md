# Describing Differences in Image or Text Sets with Natural Language

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.1-red.svg)](https://pytorch.org/get-started/previous-versions/#v21)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**This repo is a fork of [VisDiff](https://github.com/understanding-visual-datasets/VisDiff) with the text only version. It is not super clean yet.**

This repo provides the PyTorch source code of our paper: [Describing Differences in Image Sets with Natural Language](https://arxiv.org/abs/2312.02974). Check out project page [here](https://understanding-visual-datasets.github.io/VisDiff-website/)! It also contains the code for the text only version of VisDiff.

<img src="data/teaser.png"></img>

## 🚀 Getting Started

Here we provide a minimal example to describe the differences between two sets of images, where [set A](./data/examples/set_a/) are images showing `people practicing yoga in a mountainous setting` and [set B](./data/examples/set_b/) are images showing `people meditating in a mountainous setting`.

1. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

2. Login [wandb](https://wandb.ai) account:
  ```bash
  wandb login
  ```

3. Describe differences:
  ```bash
  python main.py --config configs/example.yaml
  ```

After that, you should see the following results in [wandb](https://wandb.ai/yuhuiz/VisDiff/reports/VisDiff-Example--Vmlldzo2MTUzOTk4).


## 💼 Customized Usage

If you want to use VisDiff on your own datasets, you can follow the following steps.

### 1. Convert Datasets

Convert your dataset to CSV format with two required columns `path` and `group_name`. An example of the CSV files can be found in [data/Examples.csv](data/Examples.csv).

**For text only**, replace the `path` column with `question`. It doesn't really need to be a question, but it should be text.

### 2. Define Configs

To describe the differences between two datasets, we need a `proposer` and a `ranker`. The proposer randomly samples subsets of images to generate a set of candidate differences. The ranker then scores the salience and significance of each candidate.

We have implemented different proposers and rankers in [components/proposer.py](./components/proposer.py) and [components/ranker.py](./components/ranker.py). To use each of them, you can edit arguments in [configs/base.yaml](./configs/base.yaml).

We put all the general arguments in [configs/base.yaml](./configs/base.yaml) and dataset specific arguments in [configs/example.yaml](./configs/example.yaml).

### 3. Setup Servers

We unify all the LLMs, VLMs, and CLIP to API servers for faster inference. Follow the instructions in [serve/](./serve/README.md) to start these servers.

For example, if you use BLIP-2 + GPT-4 as proposer and CLIP as ranker, you need to start the following servers:
```bash
python serve/clip_server.py
python serve/vlm_server_blip.py 
```

### 4. Describe Differences

Finally, you can run the following command to describe the differences between two datasets:
```bash
python main.py --config configs/example.yaml
```

To run the text only version:
```bash
python main.py --config configs/llm_only.yaml
```

## 🎯 Citation

If you use this repo in your research, please cite it as follows:
```
@article{VisDiff,
  title={Describing Differences in Image Sets with Natural Language},
  author={Dunlap, Lisa and Zhang, Yuhui and Wang, Xiaohan and Zhong, Ruiqi and Darrell, Trevor and Steinhardt, Jacob and Gonzalez, Joseph E. and Yeung-Levy, Serena},
  journal={arXiv preprint arXiv:2312.02974},
  year={2023}
}
```
