import copy

import numpy as np
import torch
from tianshou.data import Batch
from torch import nn

from src.policy.ppo import OptionPPOPolicy
from src.utils.common import PBE, RunningMeanStd
from src.utils.net import MLP


def get_ep_idxs(buffer):
    ep_idxs = []
    offset = 0
    for i in range(buffer.buffer_num):
        batch, idxs = buffer.buffers[i].sample(0)
        idxs_trunc = (idxs[1:] - idxs[:-1] - 1).nonzero()[0] + 1
        idxs_done = batch.done.nonzero()[0] + 1
        _ep_idxs = np.hstack([[0], idxs_trunc, idxs_done, [len(idxs)]])
        _ep_idxs = np.unique(np.hstack(_ep_idxs)) + offset
        offset += len(idxs)
        for j in range(len(_ep_idxs) - 1):
            ep_idxs.append((_ep_idxs[j], _ep_idxs[j + 1]))
    return ep_idxs


class CIM(nn.Module):
    def __init__(self, obs_dim, mlp_hidden_dims, skill_dim, proj_skill, **kwargs):
        super().__init__()
        self.encoder = MLP([obs_dim, *mlp_hidden_dims, skill_dim])
        self.proj_skill = MLP([skill_dim, *mlp_hidden_dims, skill_dim]) if proj_skill else nn.Identity()

    def forward(self, obs):
        return self.encoder(obs)

    def loss(self, obs, obs_next, option):
        key = self(obs_next) - self(obs)
        query = self.proj_skill(option.to(torch.float32))
        S = key.mm(query.T).T
        return -(S.diag() - S.logsumexp(1))


SIM_LIST = {"cim": CIM}


# Constrained Intrinsic Motivation (CIM)
class CIMPPOPolicy(OptionPPOPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.global_im_rate = cfg.p.global_im_rate
        self.intra_im = cfg.p.intra_im
        self.proj = cfg.p.proj
        self.sim_type = cfg.p.sim_type
        self.sim_rate = cfg.p.sim_rate

        self.im_name = "cim"
        assert self.rand_option or self.global_im_rate, "im is not specified"
        if self.rand_option:
            self.pbe_proj = PBE(cfg.p.k, 0, cfg.p.style)
            self._set_critic("im_proj")
            if not self.identity:
                self.sim = SIM_LIST[self.sim_type](**self.kwargs_encoder).to(self.device)
                self._set_optim(self.sim, "sim", self.im_lr)
            else:
                assert self.sim_rate == 0
                self.sim = nn.Identity()
        if self.global_im_rate:
            self.pbe_global = PBE(cfg.p.k, cfg.p.pbe_buffer_size, cfg.p.style)
            self._set_critic("im_global")

        if self.sim_rate:
            self._set_critic("im_sim")

    def process_fn(self, batch, buffer, indices):
        self.ep_idxs = np.array(get_ep_idxs(buffer))
        return super().process_fn(batch, buffer, indices)

    def _get_align_rew(self, batch, idx):
        loss = self._get_im_loss(batch, idx)
        if self.sim_type != "becl":
            return -loss
        else:
            invalid = loss == 0
            rew = (-loss).exp()
            rew[invalid] = 0
            return rew

    def _update_intrinsic_module(self, batch):
        if self.global_im_rate == 1 or self.identity:
            pass
        else:
            return super()._update_intrinsic_module(batch)

    def _get_im_loss(self, batch, idx):
        obs, obs_next = self._get_im_input(batch, idx, "both")
        option = batch.info["option"][idx]
        return self.sim.loss(obs, obs_next, option)

    def _update_im(self, loss):
        super()._update_im(loss, self.sim, "sim")

    def _get_intrinsic_rew(self, batch):
        if self.rand_option:
            proj_obs_next = self._get_empty(self.skill_dim)
            with torch.no_grad():
                for idx in self._split_batch(self.im_mbs):
                    im_input = self._get_im_input(batch, idx, "obs_next")
                    proj_obs_next[idx] = self.sim(im_input)

            if not self.identity:
                rew_sim = self._get_empty()
                with torch.no_grad():
                    for idx in self._split_batch(self.im_mbs):
                        rew_sim[idx] = self._get_align_rew(batch, idx)
                self._add_im_rew(rew_sim, batch, "im_sim")

            rew_im_proj = self._get_rew_im_proj(proj_obs_next, batch.info["option"])
            self._add_im_rew(rew_im_proj, batch, "im_proj")

        if self.global_im_rate:
            obs_next = batch.obs_next
            if self.prior_dims is not None:
                assert len(self.prior_dims) == self.skill_dim
                obs_next = obs_next[:, self.prior_dims]
            rew_im_global = self.pbe_global.get_rew(*self.pbe_global.get_xy(obs_next))
            self._add_im_rew(rew_im_global, batch, "im_global")
            self._add_im_rew(rew_im_global, batch, "cim")

    def _get_rew_im_proj(self, obs, option):
        rew_im_proj = self._get_empty()
        obs_proj = (obs * option).sum(-1, True).clamp(0)
        for idx in self.ep_idxs:
            idx = np.arange(*idx)
            x = obs_proj[idx] if self.proj else obs[idx]
            if self.intra_im:
                y = x
            else:
                y = (obs * option[idx[0]]).sum(-1, True).clamp(0) if self.proj else obs
            rew_im_proj[idx] = self.pbe_proj.get_rew(x, y)

        self.learn_info["rew_im_proj/max"] = rew_im_proj.max().item()
        self.learn_info["rew_im_proj/min"] = rew_im_proj.min().item()
        self.learn_info["rew_im_proj/mean"] = rew_im_proj.mean().item()
        return rew_im_proj

    def _get_adv_bonus(self, batch, idx):
        adv_cim = 0
        if self.rand_option:
            adv_im_proj = batch.advs["rew_im_proj"][idx]
            adv_cim = adv_im_proj
            self.learn_info["abs_adv/rew_im_proj/min"] = adv_im_proj.mean().item()
            self.learn_info["abs_adv/rew_im_proj/max"] = adv_im_proj.max().item()
            self.learn_info["abs_adv/rew_im_proj/min"] = adv_im_proj.min().item()
        if self.global_im_rate:
            adv_im_global = batch.advs["rew_im_global"][idx]
            adv_cim = (1 - self.global_im_rate) * adv_cim + self.global_im_rate * adv_im_global
            self.learn_info["abs_adv/rew_im_global"] = adv_im_global.abs().mean().item()
        if self.sim_rate:
            adv_im_sim = batch.advs["rew_im_sim"][idx]
            adv_cim += self.sim_rate * adv_im_sim
            self.learn_info["abs_adv/rew_im_sim"] = adv_im_sim.abs().mean().item()
        self.learn_info["abs_adv/rew_cim"] = adv_cim.abs().mean().item()
        return adv_cim
