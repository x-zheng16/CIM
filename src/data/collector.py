import copy
import time
from collections import defaultdict

import numpy as np
import tianshou
import torch
from fast_histogram import histogram2d
from tianshou.data import Batch, to_numpy, to_torch_as

from src.utils.common import time_spent


class Collector(tianshou.data.Collector):
    def __init__(self, policy, env, buffer, name, cfg):
        self.rand_option = cfg.p.rand_option and hasattr(policy, "gen_rand_option")
        self.switch_option = cfg.p.switch_option
        self.penalty = cfg.penalty
        self.alive_bonus = cfg.alive_bonus
        self.task_type = cfg.task_type
        self.task = cfg.task
        self.state_map = cfg.state_map
        super().__init__(policy, env, buffer, None, False)
        self.last_rew = self.last_len = self.last_success_rate = 0.0
        self.cos = torch.nn.CosineSimilarity()
        self.time_limit = cfg.time_limit
        self.name = name

    def reset_env(self, gym_reset_kwargs=None):
        super().reset_env(gym_reset_kwargs)
        self._extra_reset()

    def _reset_env_with_ids(self, local_ids, global_ids, gym_reset_kwargs=None):
        super()._reset_env_with_ids(local_ids, global_ids, gym_reset_kwargs)
        self._extra_reset(local_ids)

    def _extra_reset(self, local_ids=None):
        if self.state_map:
            for i, k in enumerate([f"{j}_position" for j in ["x", "y", "z"]]):
                if local_ids is None:
                    self.data.info[k] = self.data.obs[:, i]
                else:
                    self.data.info[k][local_ids] = self.data.obs[local_ids, i]
        if self.rand_option:
            if local_ids is None:
                option = self.policy.gen_rand_option(self.env_num)
                self.data.info["option"] = option
            else:
                option = self.policy.gen_rand_option(len(local_ids))
                self.data.info["option"][local_ids] = option

    def _update_skill(self, _ep_lens, ready_env_ids):
        if self.rand_option and self.switch_option:
            option = self.data.info["option"]
            half_ep_len = self.time_limit // 2
            switch = to_torch_as(1 - 2 * (_ep_lens == half_ep_len), option)[:, None]
            self.data.info["moss_indicator"] = (_ep_lens >= half_ep_len)[ready_env_ids]
            self.data.info["option"] = option * switch[ready_env_ids]

    def collect(self, n_step=None, n_episode=None):
        if n_step is not None:
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[: min(self.env_num, n_episode)]
        else:
            raise TypeError("Please specify at least one (either n_step or n_episode) ")

        if self.rand_option and self.name == "test" and self.policy.finetune and self.policy.solved_meta is None:
            skill_dict = defaultdict(list)
        else:
            skill_dict = None

        start_time = time.time()
        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        extra_info = defaultdict(list)
        _ep_lens = np.zeros(len(self.data.obs))
        while True:
            assert len(self.data) == len(ready_env_ids)
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            with torch.no_grad():
                result = self.policy(self.data, last_state)

            # update state / act / policy into self.data
            policy = result.get("policy", Batch())
            assert isinstance(policy, Batch)
            state = result.get("state")
            if state is not None:
                policy.hidden_state = state  # save state into buffer
            act = to_numpy(result.act)
            if self.exploration_noise:
                act = self.policy.exploration_noise(act, self.data)
            self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(action_remap, ready_env_ids)
            done = np.logical_or(terminated, truncated)
            self.data.info.update(Batch(info))
            self.data.update(obs_next=obs_next, rew=rew, terminated=terminated, truncated=truncated, done=done)
            if self.penalty is not None:
                self.data.rew -= self.penalty
            if self.alive_bonus is not None:
                self.data.rew += self.alive_bonus

            # record extra info
            if "x_velocity" in info and "y_velocity" in info:
                vel = np.stack([info[f"{k}_velocity"] for k in ["x", "y"]], axis=1)
                extra_info["vel"] += np.linalg.norm(vel, axis=-1).tolist()

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data, ready_env_ids)

            # update skill
            _ep_lens += 1
            self._update_skill(_ep_lens, ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                _ep_lens[env_ind_global] = 0

                # record useful final info
                if "x_position" in info and "y_position" in info:
                    info = self.data.info[env_ind_local]
                    pos = np.stack([info[f"{k}_position"] for k in ["x", "y"]], axis=1)
                    extra_info["r"] += np.linalg.norm(pos, axis=1).tolist()

                if skill_dict is not None:
                    for i in env_ind_local:
                        option = self.data.info["option"][i]
                        skill = "_".join(map(lambda x: str(np.round(x.item(), 1)), option))
                        skill_dict[skill].append(ep_rew[i])

                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(env_ind_local, env_ind_global)
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or (n_episode and episode_count >= n_episode):
                # print(
                #     "n_step: {}, step_count: {}, n_episode: {}, episode_count: {}".format(
                #         n_step, step_count, n_episode, episode_count
                #     )
                # )
                break

        if skill_dict is not None:
            self.policy.solve_meta(skill_dict)

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={}, act={}, rew={}, terminated={}, truncated={}, done={}, obs_next={}, info={}, policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = map(np.concatenate, [episode_rews, episode_lens, episode_start_indices])
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
            extra_infos = {}
            for k, v in extra_info.items():
                extra_infos[f"{k}"] = f"{np.mean(v):4.2f}"
            extra_infos = {k: extra_infos[k] for k in sorted(extra_infos)}
            self.last_rew, self.last_len = rew_mean, len_mean
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean, len_mean = self.last_rew, self.last_len
            rew_std = len_std = 0
            extra_infos = {}

        batch = self.buffer.sample(0)[0]
        if not batch.info.is_empty():
            success_list = batch.info[batch.done].get("success", [])
            success_rate = np.mean(success_list) if len(success_list) else self.last_success_rate
        else:
            success_list = []
            success_rate = 0
        self.last_success_rate = success_rate
        if self.task in ["Ant-v4", "Humanoid-v4", "FetchPush-v2", "FetchSlide-v2"]:
            x, y = batch.info["x_position"], batch.info["y_position"]
            H = histogram2d(x, y)
            sc = (H > 0).sum()
            extra_infos["sc"] = str(sc)
        collect_result = {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": " ".join(f"{x:.2f}" for x in rews),
            "lens": " ".join(map(str, lens)),
            # "idxs": " ".join(map(str, idxs)),
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
            "fps": int(step_count / (time.time() - start_time)),
            "time_spent": time_spent(time.time() - start_time),
            "extra_info": extra_infos,
            "success": " ".join(map(str, success_list)),
            "success_rate": float(success_rate),
        }
        return collect_result
