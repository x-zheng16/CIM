import pprint
import warnings
from os.path import join


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.simplefilter("ignore", UserWarning)
    import gymnasium
    import matplotlib
    import pkg_resources

import autoroot
import hydra
import torch
from configs.config import hydra_decorator, register_configs
from tianshou.data import VectorReplayBuffer

from src.data.collector import Collector
from src.envs.make_venv import make_venv
from src.policy import POLICY_DICT
from src.test import load_policy, test
from src.trainer.onpolicy import onpolicy_trainer
from src.utils.common import get_env_space, set_seed
from src.utils.logger import set_logger


pp = pprint.PrettyPrinter(indent=4)


@hydra.main(version_base="1.3", config_name="config")
@hydra_decorator
def train(cfg):
    # seed
    set_seed(cfg.seed)

    # logger
    logger = set_logger(cfg)
    cfg.log_dir = logger.writer.log_dir

    # env
    venv, test_venv = make_venv(cfg)
    obs_space, act_space = get_env_space(venv)

    # policy
    cfg.obs_space, cfg.act_space = obs_space, act_space
    policy = POLICY_DICT[cfg.method](cfg)
    if cfg.resume_path is not None and cfg.c.name in ["train", "debug"]:
        load_policy(cfg, policy, test_venv)

    # collector
    train_buffer = VectorReplayBuffer(cfg.c.buffer_size, len(venv))
    test_buffer = VectorReplayBuffer(cfg.c.test_buffer_size, len(test_venv))
    train_c = Collector(policy, venv, train_buffer, "train", cfg)
    test_c = Collector(policy, test_venv, test_buffer, "test", cfg)

    # train
    def save_best_fn(policy):
        # state = {"model": policy.state_dict(), "obs_rms": venv.get_obs_rms()}
        # torch.save(state, join(cfg.log_dir, "policy_best.pth"))
        pass

    def save_checkpoint_fn():
        state = {"model": policy.state_dict(), "obs_rms": venv.get_obs_rms() if cfg.p.norm_obs else None}
        torch.save(state, join(cfg.log_dir, "policy_latest.pth"))

    if not cfg.eval:
        result = onpolicy_trainer(
            policy=policy,
            train_collector=train_c,
            test_collector=test_c,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            save_state=cfg.save_state,
            state_map=cfg.state_map,
            test_after_each_epoch=cfg.test_after_each_epoch,
            verbose=cfg.verbose,
            show_progress=cfg.show_progress,
            task=cfg.task,
            rand_option=cfg.p.rand_option,
            **vars(cfg.c),
            cfg=cfg,
        )
        simple_result = {}
        keys = ["duration", "train_time/collector", "train_time/model", "train_speed"]
        for k in keys:
            simple_result.update({k: result[k]})
        pp.pprint(simple_result)

    # test
    test(cfg, policy, test_venv, test_c)

    # close
    venv.close()
    test_venv.close()


if __name__ == "__main__":
    register_configs()
    train()
