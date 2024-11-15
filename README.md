# Constrained Intrinsic Motivation for Reinforcement Learning

**Constrained Intrinsic Motivation for Reinforcement Learning (IJCAI 2024)** \[[Paper](https://arxiv.org/pdf/2407.09247)\]  
[Xiang Zheng](https://x-zheng16.github.io), [Xingjun Ma](http://xingjunma.com), Chao Shen, [Cong Wang](https://www.cs.cityu.edu.hk/~congwang/)

## Abstract

This paper investigates two fundamental problems that arise when utilizing Intrinsic Motivation (IM) for reinforcement learning in Reward-Free Pre-Training (RFPT) tasks and Exploration with Intrinsic Motivation (EIM) tasks: 1) how to design an effective intrinsic objective in RFPT tasks, and 2) how to reduce the bias introduced by the intrinsic objective in EIM tasks. Existing IM methods suffer from static skills, limited state coverage, sample inefficiency in RFPT tasks, and suboptimality in EIM tasks. To tackle these problems, we propose Constrained Intrinsic Motivation (CIM) for RFPT and EIM tasks, respectively: 1) CIM for RFPT maximizes the lower bound of the conditional state entropy subject to an alignment constraint on the state encoder network for efficient dynamic and diverse skill discovery and state coverage maximization; 2) CIM for EIM leverages constrained policy optimization to adaptively adjust the coefficient of the intrinsic objective to mitigate the distraction from the intrinsic objective. In various MuJoCo robotics environments, we empirically show that CIM for RFPT greatly surpasses fifteen IM methods for unsupervised skill discovery in terms of skill diversity, state coverage, and fine-tuning performance. Additionally, we showcase the effectiveness of CIM for EIM in redeeming intrinsic rewards when task rewards are exposed from the beginning.

## Environment

```bash
conda env create -f environment.yaml
conda activate cim
```

## Run

```bash
## test the vanilla PPO
python src/train.py -m task_type=gym task=Ant-v4 tl=200 method=base p.rf_rate=0

## unsupervised skill discovery via CIM
python src/train.py -m task_type=gym task=Ant-v4 tl=200 method=cim p.sd=2 p.ro=1 c.tt=2a7 c.spc="512*64"
```
