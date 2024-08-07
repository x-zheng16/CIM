# CIMv2

## Env Setup

```bash
conda create -n cimv2 python=3.9
conda activate cimv2

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
