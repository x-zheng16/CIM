import json

import numpy as np
import torch
from gymnasium.spaces import Box
from tianshou.data import Batch

from src.envs.wrappers.venv_wrappers import VenvWrapper
from src.utils.common import RunningMeanStd, find_all_files, get_env_space
from src.utils.net import ContinuousActor


class MazeVenv(VenvWrapper):
    def __init__(self, venv, low_actor, low_actor_obs_rms, cfg):
        super().__init__(venv)
        self.low_actor = low_actor
        self.low_actor_obs = None
        self.low_actor_obs_rms = low_actor_obs_rms
        self.skill_dim = cfg.p.skill_dim
        self.frame_skip = cfg.frame_skip
        self.option = None
        self.device = cfg.device
        self.option_scale = cfg.option_scale
        self.increment = cfg.increment
        self.exclude_agdg = cfg.exclude_agdg

        # record metadata
        self.record_metadata = cfg.rm
        self.log_dir = cfg.log_dir
        self.option_traj = []
        self.z_traj = []
        self.maze_episode_id = 0
        self.maze_env_id = 0
        self.maze_metadata = {}

    def __getattribute__(self, key):
        if key == "action_space":
            act_space = Box(low=-1, high=1, shape=(self.skill_dim,))
            return [act_space for _ in range(len(self.venv))]
        else:
            return super().__getattribute__(key)

    def step(self, act, id=None):
        id = self.venv._wrap_id(id)
        act = self._to_torch(act)
        # assert act.abs().max() <= 1
        if self.increment:
            option = self.option[id] + act * self.option_scale
            option = torch.nn.functional.normalize(option)
        else:
            option = act
        self.option[id] = option
        for _ in range(self.frame_skip):
            obs, rew, terminated, truncated, info = self._step(option, id)
        info = Batch(info)
        info["option"] = self.option[id]

        # record metadata
        if self.record_metadata:
            self.option_traj.append(self.option[self.maze_env_id].tolist())
            self.z_traj.append(obs[self.maze_env_id][4])
            done = np.logical_or(terminated, truncated)
            if done[self.maze_env_id]:
                self.maze_episode_id += 1
                meta = {"option": self.option_traj, "z": self.z_traj}
                self.maze_metadata.update({self.maze_episode_id: meta})
                self.option_traj = []
                self.z_traj = []
        return obs, rew, terminated, truncated, info

    def _step(self, option, id):
        if not self.exclude_agdg:
            obs_la = self.low_actor_obs_rms.norm(self.low_actor_obs[id, 4:])
        else:
            obs_la = self.low_actor_obs_rms.norm(self.low_actor_obs[id])
        obs_la = torch.cat([obs_la, option], 1)
        with torch.no_grad():
            act_la = self.low_actor(obs_la)[0][0]
        act_la = act_la.clamp(-1, 1).cpu().numpy()
        obs, rew, terminated, truncated, info = super().step(act_la, id)
        self.low_actor_obs[id] = self._to_torch(obs)
        return obs, rew, terminated, truncated, info

    def reset(self, id=None):
        id = self.venv._wrap_id(id)
        obs, info = super().reset(id)
        # option = torch.rand(len(id), self.skill_dim) * 2 - 1
        # option = torch.nn.functional.normalize(option).to(self.device)
        option = torch.zeros(len(obs), self.skill_dim).to(self.device)
        if self.low_actor_obs is None:
            self.low_actor_obs = self._to_torch(obs)
            self.option = option
        else:
            self.low_actor_obs[id] = self._to_torch(obs)
            self.option[id] = option
        return obs, info

    def _to_torch(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def write_metadata(self):
        if self.record_metadata and len(self.maze_metadata):
            with open(f"{self.log_dir}/option-{self.maze_episode_id}.json", "w") as f:
                json.dump(self.maze_metadata, f, indent=4)

    def close(self):
        super().close()
        self.write_metadata()


def make_venv_maze(venv, test_venv, cfg):
    obs_space, act_space = get_env_space(venv)
    cfg.obs_space, cfg.act_space = obs_space, act_space
    actor, obs_rms_th, obs_rms_np = load_maze_low_agent(cfg)
    venv = MazeVenv(venv, actor, obs_rms_th, cfg)
    test_venv = MazeVenv(test_venv, actor, obs_rms_th, cfg)
    return venv, test_venv, obs_rms_np


def load_maze_low_agent(cfg):
    if not cfg.exclude_agdg:
        obs_dim = cfg.obs_space.shape[0] - 4 + cfg.p.skill_dim
    else:
        obs_dim = cfg.obs_space.shape[0] + cfg.p.skill_dim
    act_dim = cfg.act_space.shape[0]
    actor = ContinuousActor([obs_dim] + cfg.hidden_dims + [act_dim])
    actor = actor.to(cfg.device)
    path_list = find_all_files(cfg.low_agent_path, "policy_latest.pth")
    assert len(path_list) == 1, f"find {len(path_list)} .pth files"
    print("load low_level_agent from ", path_list[0])
    ckpt = torch.load(path_list[0], map_location=cfg.device)
    state_dict = actor.state_dict()
    for k in state_dict:
        state_dict[k] = ckpt["model"]["actor." + k]
    actor.load_state_dict(state_dict)
    obs_rms_th = RunningMeanStd(device=cfg.device)
    obs_rms_th.load(ckpt["obs_rms"])
    return actor, obs_rms_th, ckpt["obs_rms"]
