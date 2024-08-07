import envpool
import gymnasium as gym
from gymnasium.wrappers import FilterObservation, FlattenObservation

from src.envs.venvs import DummyVectorEnv, ShmemVectorEnv
from src.envs.wrappers import ActionRepeat, SparseReward


GYM_TASKS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Humanoid-v4"]


def make_venv_gym(cfg):
    task, seed, n_env, n_test_env = cfg.task, cfg.seed, cfg.c.n_env, cfg.c.n_test_env
    venv_cls = DummyVectorEnv if cfg.mode in ["video", "debug"] else ShmemVectorEnv

    # common env_kwargs
    env_kwargs = {}
    is_maze = task.startswith("AntMaze") or task.startswith("PointMaze")
    is_loco = task in GYM_TASKS
    is_manipulator = task.startswith("Fetch")
    if cfg.time_limit > 0:
        env_kwargs.update({"max_episode_steps": cfg.time_limit * cfg.frame_skip})
    if is_loco:
        env_kwargs.update({"exclude_current_positions_from_observation": cfg.exclude_pos})

    # create env
    if is_loco and cfg.mode != "video":
        # specific env_kwargs
        env_kwargs.update({"seed": cfg.seed})
        if task.startswith("Ant"):
            z_min, z_max = cfg.healthy_z_range
            env_kwargs.update({"healthy_z_min": z_min, "healthy_z_max": z_max})
            env_kwargs.update({"use_contact_force": cfg.use_force})

        print("use envpool")
        venv = envpool.make_gymnasium(task, num_envs=n_env, **env_kwargs)
        test_venv = envpool.make_gymnasium(task, num_envs=n_test_env, **env_kwargs)
    else:
        # specific env_kwargs
        env_kwargs.update({"render_mode": "rgb_array"})
        if is_maze:
            env_kwargs.update(
                {
                    "continuing_task": cfg.continuing_task,
                    "camera_id": cfg.camera_id,
                    "threshold": cfg.threshold,
                    "visual": cfg.mode == "video",
                    "goal_following": cfg.goal_following,
                }
            )
        if task.startswith("Ant"):
            env_kwargs.update({"healthy_z_range": cfg.healthy_z_range})
            env_kwargs.update({"use_contact_forces": cfg.use_force})
        if is_manipulator or is_maze:
            env_kwargs.update({"pretrain": cfg.pretrain})
        print(env_kwargs)

        def env_f():
            env = gym.make(cfg.task, **env_kwargs)
            if cfg.frame_skip > 1:
                env = ActionRepeat(env, cfg.frame_skip)
            if is_maze or is_manipulator:
                if cfg.exclude_agdg:
                    env = FilterObservation(env, ["observation"])
                env = FlattenObservation(env)
            if cfg.task_type == "gym_sparse":
                env = SparseReward(env)
            return env

        venv = venv_cls([env_f for _ in range(n_env)], cfg.pid_bias, cfg.bind_core)
        test_venv = venv_cls([env_f for _ in range(n_test_env)], cfg.pid_bias, cfg.bind_core)
        venv.seed(seed)
        test_venv.seed(seed)
    # print("env_kwargs: ", env_kwargs)
    return venv, test_venv
