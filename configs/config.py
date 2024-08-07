from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, List, Optional  # noqa: UP035

import autoroot
import matplotlib as mpl
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from configs.config_collector import *  # noqa: F403
from configs.config_policy import *  # noqa: F403
from src.utils.net import MLP_NORM


mpl.use("AGG")


import matplotlib.pyplot as plt


defaults = [
    {"c": "default"},
    {"p": "default"},
    {"override hydra/job_logging": "colorlog"},
    {"override hydra/hydra_logging": "colorlog"},
    "_self_",
]

set_default = lambda x: field(default_factory=lambda: x)


@dataclass
class Config:
    c: CollectorConf = MISSING  # noqa: F405
    p: ContinuousPPOConf = MISSING  # noqa: F405

    # model
    hidden_dims: List[int] = set_default([64, 64])  # noqa: UP006
    device: str = "cuda"
    init_logstd: float = float(np.log(1))
    lr: float = 3e-4

    # logger
    mode: str = "multirun"
    resume_path: Optional[str] = None
    zz: str = ""  # comment

    # task
    task_type: str = MISSING
    task: str = MISSING
    seed: int = 0
    method: str = "base"

    # others
    show_progress: bool = False
    pid_bias: int = 0
    bind_core: bool = False
    save_state: bool = False
    state_map: bool = True
    test_after_each_epoch: bool = True
    record_traj: bool = True
    colorful_traj: bool = True
    visual_endpoint: bool = False
    eval: bool = False
    verbose: bool = True
    tl: int = MISSING  # time_limit
    penalty: Optional[float] = None
    alive_bonus: Optional[float] = None
    single_video: bool = False
    rm: bool = False  # record metadata
    save_buffer: bool = False
    visual_option: bool = False
    fps: int = 30
    use_inter_area_resize: bool = True
    use_envpool: bool = True
    override: str = ""
    resume: bool = False
    use_mix_rew: bool = False
    amplify_ex: float = 1

    # mpl setting
    wspace: float = 0
    hspace: float = 0
    style: str = "seaborn-v0_8-whitegrid"
    fig_width: float = 90
    fig_height: Optional[float] = None
    fig_title: Optional[str] = None
    xlims: Optional[List[float]] = None
    ylims: Optional[List[float]] = None
    save_pdf: bool = False
    nloc: int = 4
    fontsize: int = 16

    # representation
    use_rep_for_ac: bool = False
    use_rep_for_im: bool = False
    mlp_hidden_dims: List[int] = set_default([1024])
    obs_rep_dim: int = 64
    mlp_norm: str = "BN"

    # rew-free task kwargs
    healthy_z_range: List[float] = set_default([0.25, 1])
    exclude_pos: bool = False
    use_force: bool = False
    twu: bool = True  # terminate_when_unhealthy
    pretrain: bool = True

    # downstream task kwargs
    low_agent_path: Optional[str] = None
    frame_skip: int = 1
    threshold: float = 1
    continuing_task: bool = False
    camera_id: int = -1
    option_scale: float = 0.5
    increment: bool = True
    exclude_agdg: bool = False
    goal_following: bool = False
    state_dict_filter: Optional[str] = None
    act_scale: float = 1

    # hydra setting
    hydra: DictConfig = OmegaConf.load(to_absolute_path("configs/hydra/default.yaml"))
    defaults: List[Any] = set_default(defaults)

    def __post_init__(self):
        assert self.mlp_norm in MLP_NORM, "unsupported mlp_norm"
        eval_mode = ["eval", "video"]
        eval_method = ["zero", "random", "zeroshot", "bang"]
        if self.mode in eval_mode or self.method in eval_method:
            self.eval = True
        self.time_limit = self.tl
        self.p.pbe_buffer_size = self.p.pbe_buffer_size or self.c.buffer_size
        print("comment: ", self.zz)
        if "Maze" in self.task and self.tl > 0:
            self.penalty = 0.1 / (self.tl)  # shortest path
        if self.record_traj:
            self.c.test_buffer_size = int(
                np.ceil(self.c.episode_per_test / self.c.n_test_env) * self.c.n_test_env * self.tl
            )
        else:
            self.c.test_buffer_size = self.c.n_test_env
        self.p.im_repeat = self.p.im_repeat or self.c.repeat_per_collect
        self.p.im_mbs = self.p.im_mbs or self.c.minibatch_size

        # mlp setting
        self.fig_width = self.fig_width / 25.4
        if self.fig_height is None:
            self.fig_height = self.fig_width / 6.4 * 4.8
        if self.fig_title is None:
            self.fig_title = self.method
        plt.style.use(self.style)
        mpl.rc(["pdf", "ps"], fonttype=42)
        mpl.rc("font", size=14)
        mpl.rc("axes", titlesize=14, labelpad=0, titlepad=4)
        mpl.rc("figure.subplot", wspace=self.wspace, hspace=self.hspace)

        # finetune
        if self.mode == "finetune" and self.resume:
            root_dir = autoroot.root
            self.resume_path = self.resume_path or (
                root_dir
                / f"logs/multirun/{self.task_type}/{self.task}/{self.method}/{self.p.rs}/{self.override}/seed={self.seed}"
            )

        # goal following
        if self.goal_following:
            assert not self.pretrain, "pretrain should be False"


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    cs.store(group="c", name="default", node=CollectorConf)
    cs.store(group="c", name="debug", node=DebugCollectorConf)
    cs.store(group="c", name="eval", node=EvalCollectorConf)
    cs.store(group="c", name="video", node=VideoCollectorConf)
    cs.store(group="c", name="aprl", node=APRLCollectorConf)
    cs.store(group="p", name="default", node=ContinuousPPOConf)


def hydra_decorator(func: Callable) -> Callable:
    @wraps(func)
    def inner_decorator(cfg: DictConfig):
        cfg = OmegaConf.to_object(cfg)
        return func(cfg)

    return inner_decorator
