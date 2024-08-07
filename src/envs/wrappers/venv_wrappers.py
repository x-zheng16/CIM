import json
import os

import numpy as np
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from tianshou.data import Batch
from tianshou.env.venvs import GYM_RESERVED_KEYS, BaseVectorEnv
from tianshou.utils import RunningMeanStd

from src.envs.venvs import DummyVectorEnv


class VenvWrapper(BaseVectorEnv):
    def __init__(self, venv: BaseVectorEnv):
        self.venv = venv
        self.is_async = venv.is_async
        self.env_num = len(self)

    def __len__(self):
        return len(self.venv)

    def __getattribute__(self, key):
        if key in GYM_RESERVED_KEYS:
            return getattr(self.venv, key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(self, key, id=None):
        return self.venv.get_env_attr(key, id)

    def set_env_attr(self, key, value, id=None):
        return self.venv.set_env_attr(key, value, id)

    def reset(self, id=None, **kwargs):
        return self.venv.reset(id, **kwargs)

    def step(self, action, id=None):
        return self.venv.step(action, id)

    def seed(self, seed=None):
        return self.venv.seed(seed)

    def render(self, **kwargs):
        return self.venv.workers[0].render()

    def close(self):
        self.venv.close()

    @property
    def unwrapped(self):
        return self.venv.unwrapped


class VenvNormObs(VenvWrapper):
    def __init__(self, venv, update_obs_rms=True):
        super().__init__(venv)
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()

    def step(self, act, id=None):
        results = super().step(act, id)
        obs_original = results[0]
        info = Batch(results[-1])
        info.update({"obs_original": obs_original})
        return (self._norm_obs(obs_original), *results[1:-1], info)

    def reset(self, id=None, **kwargs):
        rval = super().reset(id, **kwargs)
        returns_info = isinstance(rval, (tuple, list)) and (len(rval) == 2)
        if returns_info:
            obs_original, info = rval
            info = Batch(info)
            info.update({"obs_original": obs_original})
        else:
            obs_original = rval
        obs = self._norm_obs(obs_original)
        return (obs, info) if returns_info else obs

    def _norm_obs(self, obs):
        if self.update_obs_rms:
            self.obs_rms.update(obs)
        return self.obs_rms.norm(obs)

    def set_obs_rms(self, obs_rms):
        self.obs_rms = obs_rms

    def get_obs_rms(self):
        return self.obs_rms


class VideoRecorder(VideoRecorder):
    def write_metadata(self):
        pass


class VenvVideoRecorder(VenvWrapper):
    def __init__(self, venv, cfg):
        super().__init__(venv)
        self.episode_id = 0
        self.task_type = cfg.task_type
        self._is_dummyvenv = isinstance(self.unwrapped, DummyVectorEnv)
        assert cfg.c.n_test_env == 1
        self.single_video = cfg.single_video
        self.log_dir = os.path.abspath(cfg.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.fps = cfg.fps
        self.meta = {"fps": self.fps, "act": "", "dist": ""}
        self.record_z = False
        self.recorded_frames = []

    def step(self, act, id=None):
        obs, rew, terminated, truncated, info = super().step(act, id)
        done = np.logical_or(terminated, truncated)

        if self._is_dummyvenv:
            frame = self.render()
            self.recorded_frames.append(frame)
        else:
            frame = obs[0][:3].transpose([1, 2, 0])
            self.recorded_frames.append(frame)

        # self.meta["act"] += f"{act[0]} "
        # self.meta["dist"] += f"{info[0].get('dist', 0)} "
        if done[0]:
            self.episode_id += 1
            # self.meta.update(
            #     {"final_obs": " ".join(f"{x:.2f}" for x in obs[0])}
            # )
            if isinstance(info, list):
                success = info[0].get("success")
                winner_id = info[0].get("winner_id")
                self.meta.update({"success": success, f"#{self.episode_id}": winner_id})
            if not self.single_video:
                self.write_videofile()
        return obs, rew, terminated, truncated, info

    @property
    def base_path(self):
        return f"{self.log_dir}/video-{self.episode_id}"

    def write_videofile(self):
        if len(self.recorded_frames) == 0:
            return

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(self.recorded_frames, fps=self.fps)
        clip.write_videofile(self.base_path + ".mp4", logger=None)
        clip.close()
        self.recorded_frames = []
        with open(self.base_path + ".json", "w") as f:
            json.dump(self.meta, f, indent=4)

    def close(self):
        super().close()
        self.write_videofile()
