from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CollectorConf:
    tt: str = "5a6"  # total_timesteps
    max_epoch: int = 20
    spc: str = "128*64"  # step_per_collect
    nmb: int = 4  # nminibatches
    n_env: int = 64
    n_test_env: int = 64
    ept: int = 1000  # episode_per_test
    repeat_per_collect: int = 10
    test_buffer_size: Optional[int] = None
    name: str = "train"

    def __post_init__(self):
        self.total_timesteps = int(eval(self.tt.replace("a", "e")))
        self.step_per_collect = int(eval(self.spc.replace("a", "e")))
        self.nminibatches = self.nmb
        self.episode_per_test = self.ept

        self.step_per_epoch = self.total_timesteps // self.max_epoch
        self.buffer_size = self.step_per_collect
        self.minibatch_size = self.step_per_collect // self.nminibatches
        self.total_updates = np.ceil(self.step_per_epoch / self.step_per_collect) * self.max_epoch


@dataclass
class DebugCollectorConf(CollectorConf):
    tt: str = "3a3"
    max_epoch: int = 3
    spc: str = "1024"
    n_env: int = 3
    n_test_env: int = 3
    ept: int = 3
    name: str = "debug"


@dataclass
class EvalCollectorConf(CollectorConf):
    n_env: int = 1
    name: str = "eval"


@dataclass
class VideoCollectorConf(CollectorConf):
    n_env: int = 1
    n_test_env: int = 1
    ept: int = 3
    name: str = "video"


APRL_CONF = dict(
    total_timesteps=int(20e6), batch_size="2048*8", learning_rate=3e-4, nminibatches=4, noptepochs=4, num_env=8
)


@dataclass
class APRLCollectorConf(CollectorConf):
    tt: str = "2a7"
    spc: str = APRL_CONF["batch_size"]
    nmb: int = APRL_CONF["nminibatches"]
    n_env: int = APRL_CONF["num_env"]
    repeat_per_collect: int = APRL_CONF["noptepochs"]
