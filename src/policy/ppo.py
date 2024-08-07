import copy
from collections import defaultdict

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tianshou.policy.base import _gae_return
from tianshou.utils import MovAvg, RunningMeanStd
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.utils.clip_grad import clip_grad_norm_

from src.utils.common import get_dist_fn, grad_monitor, last_layer_init, param_monitor, split_batch, weight_init
from src.utils.net import get_actor_critic, get_encoder


# Proximal Policy Optimization (PPO)
class PPOPolicy(BasePolicy):
    def __init__(self, cfg):
        # basic
        super().__init__(cfg.obs_space, cfg.act_space, cfg.p.action_scaling, cfg.p.action_bound_method)

        # model params
        self.use_rep_for_ac = cfg.use_rep_for_ac
        self.use_rep_for_im = cfg.use_rep_for_im
        self.device = cfg.device
        self.obs_shape = cfg.obs_space.shape
        self.act_shape = cfg.act_space.shape or cfg.act_space.n
        self.act_dim = np.prod(self.act_shape)
        print(f"obs_shape: {self.obs_shape} | act_shape: {self.act_shape}")
        self.obs_rep_dim = cfg.obs_rep_dim
        self.obs_dim = self.obs_rep_dim if self.use_rep_for_im else self.obs_shape[0]
        self.mlp_hidden_dims = cfg.mlp_hidden_dims
        self.mlp_norm = cfg.mlp_norm

        if cfg.p.prior_dims is not None:
            if cfg.p.prior_dims[0] < 0:
                self.prior_dims = np.arange(self.obs_dim)
                self.prior_dims = np.delete(self.prior_dims, -np.array(cfg.p.prior_dims))
            else:
                self.prior_dims = cfg.p.prior_dims
            print("prior_dims is ", self.prior_dims)
        else:
            self.prior_dims = None

        self.use_prior_as_input = cfg.p.use_prior_as_input
        self.identity = cfg.p.identity
        self.kwargs_encoder = {
            "obs_shape": self.obs_shape,
            "obs_dim": self.obs_dim if not self.use_prior_as_input else len(self.prior_dims),
            "act_dim": self.act_dim,
            "act_type": self.action_type,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "obs_rep_dim": self.obs_rep_dim,
            "mlp_norm": self.mlp_norm,
            "device": self.device,
            "use_rep_for_im": self.use_rep_for_im,
            "identity": self.identity,
            "scale": cfg.p.scale,
        }
        self.hidden_dims = cfg.hidden_dims

        # models
        self.encoder = get_encoder(**self.kwargs_encoder) if self.use_rep_for_ac or self.use_rep_for_im else None
        self._set_ac_input_dim()
        self.actor, self.critic = get_actor_critic(
            self.ac_input_dim, self.hidden_dims, self.act_dim, self.action_type, device=self.device
        )
        self.critics = {"rew_ex": self.critic}

        # init
        if hasattr(self.actor, "logstd"):
            nn.init.constant_(self.actor.logstd, cfg.init_logstd)
        if self.action_type == "continuous":
            last_layer_init(self.actor)
        last_layer_init(self.critic)

        # optim
        self.lr = cfg.lr
        self.optims = {}
        self._set_optim(self.encoder, "encoder")
        self._set_optim(self.critic, "rew_ex")
        self.optim = torch.optim.Adam(self.actor.parameters(), cfg.lr)

        # ppo params
        self._eps_clip = cfg.p.eps_clip
        self._norm_adv = cfg.p.norm_adv
        self._recompute_adv = cfg.p.rc_adv
        self._grad_norm = cfg.p.max_grad_norm
        self._gae_lambda = cfg.p.gae_lambda
        self._gamma = cfg.p.discount_factor
        self._norm_return = cfg.p.norm_return
        self._deterministic_eval = cfg.p.deterministic_eval
        self._eps = 1e-8
        self.minibatch_size = cfg.c.minibatch_size
        self.learn_info = defaultdict(list)

        # others
        self.dist_fn = get_dist_fn(cfg.act_space)
        self.mse = nn.MSELoss(reduction="none")
        self.ret_rmss = defaultdict(RunningMeanStd)
        self.max_episode_steps = cfg.tl
        self.im_name = None
        self.use_mix_rew = cfg.use_mix_rew
        self.amplify_ex = cfg.amplify_ex
        if self.use_mix_rew:
            self._set_critic("mix")
            self._set_optim(self.critics["rew_mix"], "rew_mix")

        # rew schedule
        self.rew_schedule = cfg.p.rew_schedule
        self.rf_rate = cfg.p.rf_rate
        self.ex_rate = cfg.p.ex_rate
        self.lag_rate = cfg.p.lag_rate
        self.total_updates = cfg.c.total_updates
        self.last_avg_ex_rew = MovAvg(1)
        self.expected_avg_ex_rew = 0
        self.n_update, self.progress, self.lag_coef, self.rew_coef = 0, 0, 0, 0

        # encoder
        self.encoder_repeat = cfg.p.encoder_repeat
        self.encoder_mbs = cfg.p.encoder_mbs

        # finetune
        self.finetune = cfg.mode == "finetune"
        if cfg.p.solved_meta is not None:
            self.solved_meta = torch.tensor(
                list(map(int if self.is_one_hot else float, cfg.p.solved_meta.split("_")))
            ).to(self.device)
            print("solved_meta is", self.solved_meta)
        else:
            self.solved_meta = None

    def _set_ac_input_dim(self):
        self.ac_input_dim = self.obs_rep_dim if self.use_rep_for_ac else self.obs_shape[0]

    def _set_critic(self, name):
        name = "rew_" + name
        self.critics[name] = copy.deepcopy(self.critic)
        self.critics[name].apply(weight_init)
        last_layer_init(self.critics[name])
        self._set_optim(self.critics[name], name)

    def _unset_critic(self, name):
        name = "rew_" + name
        self.critics.pop(name)
        self.optims.pop(name)

    def forward(self, batch, state=None, **kwargs):
        return self._get_actor_output(self._get_mlp_input(batch), state, **kwargs)

    def _get_obs_rep(self, obs):
        return self.encoder(self._to_torch(obs))

    def _get_mlp_input(self, batch, idx=None, mode="obs", name="ac"):
        idx = self._wrap_idx(batch, idx)
        if name == "ac":
            use_rep = self.use_rep_for_ac
        elif name == "im":
            use_rep = self.use_rep_for_im
        else:
            raise Exception("unsupported mlp name")
        assert mode in ["obs", "obs_next", "both"], "unsupported input mode"
        if mode in ["obs", "both"]:
            obs = self._to_torch(batch.obs[idx])
            obs = self._get_obs_rep(obs) if use_rep else obs
        if mode in ["obs_next", "both"]:
            obs_next = self._to_torch(batch.obs_next[idx])
            obs_next = self._get_obs_rep(obs_next) if use_rep else obs_next
        if mode == "obs":
            return obs
        if mode == "obs_next":
            return obs_next
        if mode == "both":
            return obs, obs_next

    def _wrap_idx(self, batch, idx):
        return list(range(len(batch))) if idx is None else idx

    def _logits_to_dist(self, logits):
        if isinstance(logits, tuple) or isinstance(logits, list):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        return dist

    def _get_actor_output(self, ac_input, state=None, **kwargs):
        logits, state = self.actor(ac_input, state=state)
        dist = self._logits_to_dist(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        if self.action_type == "continuous":
            logits = {str(i): logits[i] for i in range(len(logits))}
        return Batch(act=act, state=state, dist=dist, policy=Batch(logits=logits))

    # process before learn
    def process_fn(self, batch, buffer, indices):
        self.learn_info.clear()
        self._buffer, self._indices, self._batch_size = buffer, indices, len(batch)
        batch.v_s, batch.returns, batch.advs = {}, {}, {}
        batch.act = self._to_torch(batch.act)
        if len(self.obs_shape) == 1:  # state-based
            batch.obs = self._to_torch(batch.obs)
            batch.obs_next = self._to_torch(batch.obs_next)
        for k in [f"{i}_{j}" for i in ["x", "y", "z"] for j in ["position", "position_prev"]]:
            if k in batch.info:
                batch.info[k] = self._to_torch(batch.info[k])
        if "rew_ex" not in batch.info:
            batch.info["rew_ex"] = batch.rew
        old_log_prob = self._get_empty()
        with torch.no_grad():
            for idx in self._split_batch():
                old_log_prob[idx] = self(batch[idx]).dist.log_prob(batch.act[idx])
        batch.logp_old = old_log_prob
        return batch

    def learn(self, batch, minibatch_size, repeat):
        learn_info = self.learn_info
        learn_info["batch"] = batch
        # encoder
        if self.encoder is not None:
            self._update_encoder(batch)

        # intrinsic module
        if not self.finetune:
            self._update_intrinsic_module(batch)
            self._get_intrinsic_rew(batch)
            self._update_rew_coef(batch)

        # actor-critic
        for step in range(repeat):
            if self._recompute_adv or step == 0:
                if self.finetune:
                    critics_name = ["rew_ex"]
                else:
                    if self.use_mix_rew:
                        if self.im_name is not None:
                            rew_im = batch.info["rew_" + self.im_name]
                        else:
                            rew_im = np.zeros_like(batch.info["rew_ex"])
                        batch.info["rew_mix"] = batch.info["rew_ex"] * self.amplify_ex + rew_im * self.rew_coef
                        critics_name = ["rew_mix"]
                    else:
                        critics_name = set(self.critics) - set("rew_mix")
                self._compute_returns(batch, critics_name)

            for idx in self._split_batch(minibatch_size, shuffle=True):
                with torch.no_grad():
                    ac_input = self._get_mlp_input(batch, idx)
                self._learn_actor(batch, idx, ac_input)
                self._learn_critics(batch, idx, ac_input)
        self.n_update += 1
        self.progress = min(self.n_update / self.total_updates, 1)
        learn_info["progress"] = self.progress
        return learn_info

    def _update_encoder(self, batch):
        for _ in range(self.encoder_repeat):
            for idx in self._split_batch(self.encoder_mbs, shuffle=True):
                loss = self.encoder.loss(self._to_torch(batch.obs[idx]))

                self.optims["encoder"].zero_grad()
                loss.backward()
                self._clip_grad(self.encoder)
                self.optims["encoder"].step()
                self.learn_info["loss/encoder"].append(loss.item())
                self.learn_info["grad/encoder"].append(grad_monitor(self.encoder))

    def _get_intrinsic_rew(self, batch):
        pass

    def _update_intrinsic_module(self, batch):
        pass

    def _update_rew_coef(self, batch):
        if self.rew_schedule == "L":
            last_avg_ex_rew = self.last_avg_ex_rew.get()
            self.learn_info["last_avg_ex_rew"] = last_avg_ex_rew
            ex_rew = batch.info["rew_ex"][batch.done]
            avg_ex_rew = np.mean(ex_rew) if len(ex_rew) else last_avg_ex_rew
            self.expected_avg_ex_rew = max(min(self.ex_rate * last_avg_ex_rew, 1), self.expected_avg_ex_rew)
            self.learn_info["expected_avg_ex_rew"] = self.expected_avg_ex_rew
            grad_lag = avg_ex_rew - self.expected_avg_ex_rew
            self.learn_info["grad_lag"] = grad_lag
            self.lag_coef = max(self.lag_coef - self.lag_rate * grad_lag, 0)
            self.last_avg_ex_rew.add(avg_ex_rew)
            self.rew_coef = 1 / (1 + self.lag_coef)
            self.learn_info["lag_coef"] = self.lag_coef
        elif self.rew_schedule == "PLinear":
            self.rew_coef = 1 - self.progress
        elif self.rew_schedule == "PConst":
            self.rew_coef = 1
        elif self.rew_schedule == "PExp":
            self.rew_coef = 0.001**self.progress
        self.learn_info["rew_coef"] = self.rew_coef

    def _learn_actor(self, batch, idx, ac_input):
        learn_info = self.learn_info

        dist = self._get_actor_output(ac_input).dist
        adv = self._get_adv(batch, idx)
        adv_mean, adv_std = adv.mean(), adv.std()
        learn_info["adv/mean"].append(adv_mean.item())
        learn_info["adv/std"].append(adv_std.item())
        adv = (adv - adv_mean) / (adv_std + 1e-10) if self._norm_adv else adv
        ratio = (dist.log_prob(batch.act[idx]) - batch.logp_old[idx]).exp().float()
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * adv
        clip_loss = -torch.min(surr1, surr2).mean()

        self.optim.zero_grad()
        clip_loss.backward()
        self._clip_grad(self.actor)
        self.optim.step()

        learn_info["loss/actor"].append(clip_loss.item())
        learn_info["grad/actor"].append(grad_monitor(self.actor))
        learn_info["param/actor"].append(param_monitor(self.actor.mean.mlp))
        with torch.no_grad():
            approx_kl = (ratio - 1) - ratio.log()
            clip_frac = ((ratio - 1).abs() > self._eps_clip).float().mean()
        learn_info["max_ratio"].append(ratio.max().item())
        learn_info["max_kl"].append(approx_kl.max().item())
        learn_info["mean_kl"].append(approx_kl.mean().item())
        learn_info["clip_frac"].append(clip_frac.item())

    def _learn_critics(self, batch, idx, ac_input):
        if self.finetune:
            critics_name = ["rew_ex"]
        else:
            if self.use_mix_rew:
                critics_name = ["rew_mix"]
            else:
                critics_name = set(self.critics) - set("rew_mix")

        for k in critics_name:
            vf_loss = self.mse(self.critics[k](ac_input), batch.returns[k][idx]).mean()

            self.optims[k].zero_grad()
            vf_loss.backward()
            self._clip_grad(self.critics[k])
            self.optims[k].step()

            learn_info = self.learn_info
            learn_info["loss/critic/" + k].append(vf_loss.item())
            learn_info["grad/critic/" + k].append(grad_monitor(self.critics[k]))

    def _get_adv(self, batch, idx):
        if self.use_mix_rew:
            return batch.advs["rew_mix"][idx]
        adv_ex = batch.advs["rew_ex"][idx]
        self.learn_info["abs_adv_ex"] = adv_ex.abs().mean().item()
        if self.finetune:
            return adv_ex

        adv_bonus = self._get_adv_bonus(batch, idx)
        if self.rew_schedule == "RF":
            return adv_bonus if self.progress < self.rf_rate else adv_ex
        else:
            assert self.rew_schedule in ["L", "PConst", "PLinear", "PExp"]
            return adv_ex + self.rew_coef * adv_bonus

    def _get_adv_bonus(self, batch, idx):
        if self.im_name is not None:
            adv_bonus = batch.advs["rew_" + self.im_name][idx]
            self.learn_info["abs_adv_bonus"] = adv_bonus.abs().mean().item()
        else:
            adv_bonus = torch.zeros_like(batch.advs["rew_ex"][idx])
        return adv_bonus

    def _compute_returns(self, batch, critics_name):
        v_s = defaultdict(self._get_empty)
        v_s_next = defaultdict(self._get_empty)
        with torch.no_grad():
            for idx in self._split_batch():
                ac_input, ac_input_next = self._get_mlp_input(batch, idx, "both", "ac")
                for k in critics_name:
                    v_s[k][idx] = self.critics[k](ac_input)
                    v_s_next[k][idx] = self.critics[k](ac_input_next)
        batch.v_s = v_s
        for k in critics_name:
            batch.returns[k], batch.advs[k] = self.__compute_returns(
                batch, v_s_next[k].cpu().numpy(), v_s[k].cpu().numpy(), self.ret_rmss[k], batch.info[k]
            )
            xvar = 1 - (batch.returns[k] - batch.v_s[k]).var() / (batch.returns[k].var() + 1e-8)
            self.learn_info[f"xvar/{k}"].append(xvar.item())

    def __compute_returns(self, batch, v_s_next, v_s, ret_rms, rew):
        if self._norm_return:
            v_s = v_s * np.sqrt(ret_rms.var + self._eps)
            v_s_next = v_s_next * np.sqrt(ret_rms.var + self._eps)

        unnormalized_returns, advantages = self._compute_episodic_return(batch, v_s_next, v_s, rew)
        if self._norm_return:
            returns = unnormalized_returns / np.sqrt(ret_rms.var + self._eps)
            ret_rms.update(unnormalized_returns)
        else:
            returns = unnormalized_returns
        returns = self._to_torch(returns)
        advantages = self._to_torch(advantages)
        return returns, advantages

    def _compute_episodic_return(self, batch, v_s_next, v_s, rew):
        buffer, indices = self._buffer, self._indices
        v_s_next = v_s_next * BasePolicy.value_mask(buffer, indices)
        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_next, rew, end_flag, self._gamma, self._gae_lambda)
        returns = advantage + v_s
        return returns, advantage

    def _to_torch(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _to_numpy(self, x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def _get_empty(self, *shape):
        return torch.empty([self._batch_size, *shape], device=self.device)

    def _split_batch(self, minibatch_size=None, batch_size=None, shuffle=False):
        return split_batch(minibatch_size or self.minibatch_size, batch_size or self._batch_size, shuffle)

    def _clip_grad(self, net):
        if self._grad_norm is not None and net is not None:
            clip_grad_norm_(net.parameters(), self._grad_norm)

    def _set_optim(self, net, optim_name, lr=None):
        try:
            optim = torch.optim.Adam(net.parameters(), lr or self.lr)
        except:
            print(f"use fake optim for {optim_name}")
            optim = torch.optim.Adam([nn.Parameter(torch.zeros(1))])
        self.optims[optim_name] = optim


# PPOPolicy with Option
class OptionPPOPolicy(PPOPolicy):
    def __init__(self, cfg):
        self.skill_dim = cfg.p.skill_dim
        self.rand_option = cfg.p.rand_option
        self._solve_meta = cfg.p.solve_meta
        self.is_one_hot = cfg.p.is_one_hot
        super().__init__(cfg)
        self.kwargs_encoder.update(
            {
                "skill_dim": self.skill_dim,
                "diff_state": cfg.p.diff_state,
                "temperature": cfg.p.temp,
                "proj_skill": cfg.p.proj_skill,
            }
        )
        self.im_repeat = cfg.p.im_repeat
        self.im_mbs = cfg.p.im_mbs
        self.im_lr = cfg.p.im_lr

    def _set_ac_input_dim(self):
        super()._set_ac_input_dim()
        self.ac_input_dim += self.skill_dim if self.rand_option else 0

    def _get_mlp_input(self, batch, idx=None, mode="obs", name="ac"):
        idx = self._wrap_idx(batch, idx)
        result = super()._get_mlp_input(batch, idx, mode, name)
        if name == "ac" and self.rand_option:
            option = batch.info["option"][idx]
            if isinstance(result, tuple):
                obs, obs_next = result
                obs = torch.cat([obs, option], 1)
                obs_next = torch.cat([obs_next, option], 1)
                result = (obs, obs_next)
            else:
                result = torch.cat([result, option], 1)
        return result

    def gen_rand_option(self, batch_size):
        if self.solved_meta is not None:
            option = self.solved_meta.repeat((batch_size, 1))
        else:
            if not self.is_one_hot:
                option = torch.rand(batch_size, self.skill_dim) * 2 - 1
                if not self.training:
                    option = nn.functional.normalize(option)
            else:
                option = torch.randint(0, self.skill_dim, (batch_size,))
                option = one_hot(option, self.skill_dim)
        return option.to(self.device)

    def _get_im_input(self, batch, idx=None, mode="obs"):
        idx = self._wrap_idx(batch, idx)
        if not self.use_prior_as_input:
            with torch.no_grad():
                return self._get_mlp_input(batch, idx, mode, "im")
        else:
            assert len(self.obs_shape) == 1
            obs = batch.obs[idx][:, self.prior_dims]
            obs_next = batch.obs_next[idx][:, self.prior_dims]
            if mode == "obs":
                return obs
            if mode == "obs_next":
                return obs_next
            if mode == "both":
                return obs, obs_next

    def _get_intrinsic_rew(self, batch):
        rew_im = self._get_empty()
        with torch.no_grad():
            for idx in self._split_batch():
                rew_im[idx] = self._get_im_rew(batch, idx)
        self._add_im_rew(rew_im, batch)

    def _get_im_rew(self, batch, idx):
        pass

    def _add_im_rew(self, rew_im, batch, rew_name=None):
        rew_name = "rew_" + rew_name if rew_name is not None else "rew_" + self.im_name
        rew_im = self._to_numpy(rew_im)
        batch.info[rew_name] = rew_im
        self.learn_info[rew_name] = rew_im.mean()

    def _update_intrinsic_module(self, batch):
        for _ in range(self.im_repeat):
            for idx in self._split_batch(self.im_mbs, shuffle=True):
                loss = self._get_im_loss(batch, idx).mean()
                self._update_im(loss)

    def _get_im_loss(self, batch, idx):
        pass

    def _update_im(self, loss, im_module=None, im_name=None):
        if not self.identity and im_module is not None and im_name is not None:
            self.optims[im_name].zero_grad()
            loss.backward()
            self._clip_grad(im_module)
            self.optims[im_name].step()
            self.learn_info[f"loss/{im_name}"].append(loss.item())
            self.learn_info[f"grad/{im_name}"].append(grad_monitor(im_module))

    def solve_meta(self, skill_dict):
        print("=" * 40)
        if self._solve_meta:
            max_ep_rew = -np.inf
            best_skill = None
            for k, v in skill_dict.items():
                mean_ep_rew = np.mean(v)
                if mean_ep_rew > max_ep_rew:
                    print(k, v)
                    max_ep_rew = mean_ep_rew
                    best_skill = k
            self.solved_meta = torch.tensor(list(map(int if self.is_one_hot else float, best_skill.split("_")))).to(
                self.device
            )
            print("solved_skill is ", self.solved_meta, "max_ep_rew is ", max_ep_rew)
        else:
            self.solved_meta = self.gen_rand_option(1)[0]
            print("randomly generate a fix skill", self.solved_meta)
        print("=" * 40)
