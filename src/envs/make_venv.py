from src.envs.env_atari import make_venv_atari
from src.envs.env_gym import make_venv_gym
from src.envs.env_maze import make_venv_maze
from src.envs.env_sort import make_venv_sort


try:
    from src.envs.env_vizdoom import make_venv_vizdoom

    VIZDOOM = True
except:  # noqa: E722
    VIZDOOM = False
from src.envs.wrappers.venv_wrappers import VenvNormObs, VenvVideoRecorder


def make_venv(cfg):
    if cfg.task_type == "vizdoom" and VIZDOOM:
        venv, test_venv = make_venv_vizdoom(cfg)
    elif cfg.task_type == "atari":
        venv, test_venv = make_venv_atari(cfg)
    elif cfg.task_type == "sort":
        venv, test_venv = make_venv_sort(cfg)
    else:
        venv, test_venv = make_venv_gym(cfg)

    if cfg.mode == "video":
        test_venv = VenvVideoRecorder(test_venv, cfg)
    if cfg.task_type == "meta_maze":
        venv, test_venv, _ = make_venv_maze(venv, test_venv, cfg)
    if cfg.p.norm_obs:
        assert cfg.task_type not in ["vizdoom", "atari"]
        venv = VenvNormObs(venv)
        test_venv = VenvNormObs(test_venv, update_obs_rms=False)
        test_venv.set_obs_rms(venv.get_obs_rms())
    return venv, test_venv
