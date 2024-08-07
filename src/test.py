import json
import pprint
from os.path import join

import torch

from src.utils.common import find_all_files


pp = pprint.PrettyPrinter(indent=4)


def load_policy(cfg, policy, test_venv):
    print("Load agent from: ", cfg.resume_path)
    path_list = find_all_files(cfg.resume_path, "policy_latest.pth")
    print(path_list)
    assert len(path_list) == 1, f"find {len(path_list)} .pth files"
    ckpt = torch.load(path_list[0], map_location=cfg.device)
    state_dict = policy.state_dict()
    if cfg.state_dict_filter is not None:
        state_dict.update(
            {k: v for k, v in ckpt["model"].items() if k in state_dict and k.startswith(cfg.state_dict_filter)}
        )
    else:
        state_dict.update({k: v for k, v in ckpt["model"].items() if k in state_dict})
    policy.load_state_dict(state_dict)
    obs_rms = ckpt["obs_rms"]
    if obs_rms is not None and cfg.p.norm_obs:
        test_venv.set_obs_rms(obs_rms)
    if cfg.method == "zeroshot":
        policy.set_obs_rms(obs_rms)
    print("Load agent from: ", path_list[0])


def test(cfg, policy, test_venv, test_c):
    if cfg.resume_path is not None and cfg.c.name in ["eval", "video"]:
        load_policy(cfg, policy, test_venv)

    # eval
    policy.eval()
    test_c.reset()
    n_episode = cfg.c.episode_per_test
    result = test_c.collect(n_episode=n_episode)
    time = f"{test_c.collect_time:.2f} s"
    fps = f"{test_c.collect_step / test_c.collect_time:.2f} step/s"
    result.update({"test_time": time, "test_fps": fps})

    # save
    with open(join(cfg.log_dir, f"result_{n_episode}.json"), "w") as f:
        json.dump(result, f, indent=4)
    if cfg.save_buffer:
        torch.save(test_c.buffer, join(cfg.log_dir, "test_buffer.pkl"))
    keys = ["n/ep", "rew", "rew_std", "len", "len_std", "success_rate", "test_time"]
    pp.pprint({k: v for k, v in result.items() if k in keys})
