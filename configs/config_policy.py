from dataclasses import dataclass
from typing import Optional


REW_SCHEDULE = ["RF", "PConst", "PLinear", "PExp", "L"]
REW_TYPE = ["v"]


@dataclass
class ContinuousPPOConf:
    # basic
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    eps_clip: float = 0.2
    norm_obs: bool = True
    norm_return: bool = True
    norm_adv: bool = True
    rc_adv: bool = False
    action_bound_method: str = "clip"
    action_scaling: bool = True
    deterministic_eval: bool = True

    # common
    prior_dims: Optional[list[int]] = None
    use_prior_as_input: Optional[bool] = False
    sd: int = 0  # skill dim
    ro: bool = False  # gen random option
    so: bool = False  # switch option
    is_one_hot: bool = False  # decoupled option
    ds: bool = False  # diff_state style of im loss
    sim_type: str = "cim"  # similarity loss for im
    im_repeat: Optional[int] = 10
    im_mbs: Optional[int] = 128
    im_lr: Optional[float] = 3e-4
    encoder_repeat: Optional[int] = 10
    encoder_mbs: Optional[int] = 128
    num_stack: int = 1
    scale: bool = True
    solve_meta: bool = False
    solved_meta: Optional[str] = None

    # pbe
    pbs: Optional[str] = None  # buffer size for entropy estimator
    k: int = 10  # k-nearest neighbour
    style: str = "log_mean"

    # cim
    sim_rate: float = 0
    gi_rate: float = 0  # global_im_rate
    intra_im: bool = True  # intra-trajectory or not
    proj: bool = True

    # visr, vic
    ie: bool = False  # identity encoder for intrinsic module

    # cic, becl, moss
    temp: float = 0.5  # temperature
    proj_skill: bool = False

    # dense
    rew_type: Optional[str] = None  # dense reward type

    # rew schdule
    rs: str = "RF"
    rf_rate: float = 1  # determining proportion of reward-free pre-training stage [0,1]
    ex_rate: float = 1  # estimating upper_bound of last_ex_return [1,inf)
    lag_rate: float = 10  # update step of Lagrangian coefficient

    def __post_init__(self):
        if self.pbs is not None:
            self.pbs = int(eval(self.pbs.replace("a", "e")))
        self.pbe_buffer_size = self.pbs
        assert self.rs in REW_SCHEDULE, "unsupported rew schedule"
        self.rew_schedule = self.rs
        self.identity = self.ie
        self.skill_dim = self.sd
        self.rand_option = self.ro
        self.switch_option = self.so
        self.diff_state = self.ds
        self.global_im_rate = self.gi_rate
        if self.global_im_rate == 1:
            assert not self.rand_option
        if self.use_prior_as_input:
            assert self.prior_dims is not None
