# Constrained Intrinsic Motivation for Reinforcement Learning

**Constrained Intrinsic Motivation for Reinforcement Learning (IJCAI 2024)** \[[Paper](https://arxiv.org/pdf/2407.09247)\]  
[Xiang Zheng](https://x-zheng16.github.io), [Xingjun Ma](http://xingjunma.com), Chao Shen, [Cong Wang](https://www.cs.cityu.edu.hk/~congwang/)

## Abstract

This paper investigates two fundamental problems that arise when utilizing Intrinsic Motivation (IM) for reinforcement learning in Reward-Free Pre-Training (RFPT) tasks and Exploration with Intrinsic Motivation (EIM) tasks: 1) how to design an effective intrinsic objective in RFPT tasks, and 2) how to reduce the bias introduced by the intrinsic objective in EIM tasks. Existing IM methods suffer from static skills, limited state coverage, sample inefficiency in RFPT tasks, and suboptimality in EIM tasks. To tackle these problems, we propose Constrained Intrinsic Motivation (CIM) for RFPT and EIM tasks, respectively: 1) CIM for RFPT maximizes the lower bound of the conditional state entropy subject to an alignment constraint on the state encoder network for efficient dynamic and diverse skill discovery and state coverage maximization; 2) CIM for EIM leverages constrained policy optimization to adaptively adjust the coefficient of the intrinsic objective to mitigate the distraction from the intrinsic objective. In various MuJoCo robotics environments, we empirically show that CIM for RFPT greatly surpasses fifteen IM methods for unsupervised skill discovery in terms of skill diversity, state coverage, and fine-tuning performance. Additionally, we showcase the effectiveness of CIM for EIM in redeeming intrinsic rewards when task rewards are exposed from the beginning.

## Environments

```bash
conda create -n cim python=3.9
conda activate cim

pip install -U pip
pip install ipykernel ipywidgets black isort setuptools autoroot
pip install hydra-core hydra-colorlog fast-histogram pykeops
pip install seaborn matplotlib
pip install tianshou==0.5.0
pip install mujoco==2.3.3
pip install gymnasium=0.28.1
pip install envpool opencv-python
apt install cmake git libboost-all-dev libsdl2-dev libopenal-dev
pip install vizdoom
```
