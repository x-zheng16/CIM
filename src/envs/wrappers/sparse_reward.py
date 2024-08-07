from gymnasium import Wrapper

SPARSE_TASK = {
    "Walker2d-v3": ("x", 5),
    "HalfCheetah-v3": ("x", 20),
    "Ant-v4": ("x", 60),
}


class SparseReward(Wrapper):
    def step(self, act):
        obs, rew_original, _, truncated, info = super().step(act)
        if self.spec.id in SPARSE_TASK:
            k, goal = SPARSE_TASK[self.spec.id]
            success = info[f"{k}_position"] >= goal
            rew_dist = info["x_position"] - goal
            rew_vel = info["x_velocity"]
        elif self.spec.id.startswith("Reacher"):
            success = (-info["reward_dist"]) <= 0.005
            rew_dist = info["reward_dist"]
            rew_vel = 0
        is_healthy = getattr(self, "is_healthy", True)
        terminated = success or (not is_healthy)
        rew = success
        rew -= 0.1 / self.spec.max_episode_steps
        info.update(
            {
                "is_healthy": is_healthy,
                "rew_original": rew_original,
                "rew_dist": rew_dist,
                "rew_vel": rew_vel,
                "success": success,
            }
        )
        return obs, rew, terminated, truncated, info
