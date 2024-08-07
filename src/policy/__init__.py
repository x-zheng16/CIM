import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy

from src.policy.cim import CIMPPOPolicy
from src.policy.ppo import PPOPolicy


POLICY_DICT = {"cim": CIMPPOPolicy, "base": PPOPolicy}


class RandomPolicy(BasePolicy):
    def __init__(self, cfg):
        super().__init__(cfg.obs_space, cfg.act_space, cfg.p.action_scaling, cfg.p.action_bound_method)
        self.act_shape = cfg.act_space.shape or cfg.act_space.n
        self.act_dim = np.prod(self.act_shape)
        self.action_space.seed(cfg.seed)

    def forward(self, batch, state=None, **kwargs):
        if self.action_type == "discrete":
            act = [self.action_space.sample() for _ in range(len(batch))]
        else:
            act = torch.rand(len(batch), self.act_dim) * 2 - 1
        return Batch(act=act)

    def learn(self, batch, **kwargs):
        pass


class ZeroPolicy(RandomPolicy):
    def forward(self, batch, state=None, **kwargs):
        act = torch.zeros(len(batch), self.act_dim)
        return Batch(act=act)


POLICY_DICT.update({"random": RandomPolicy, "zero": ZeroPolicy})
