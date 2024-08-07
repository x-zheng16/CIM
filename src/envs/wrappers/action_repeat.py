from gymnasium import Wrapper


class ActionRepeat(Wrapper):
    def __init__(self, env, frame_skip=1):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, act):
        for _ in range(self.frame_skip):
            obs, rew, terminated, truncated, info = super().step(act)
        return obs, rew, terminated, truncated, info
