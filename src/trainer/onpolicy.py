import inspect
import time
from collections import defaultdict

import torch
from tianshou.trainer import BaseTrainer

from src.data.collector import Collector
from src.utils.common import LogIt, time_remain
from src.utils.logger import Logger


class OnpolicyTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        self.minibatch_size = kwargs.get("minibatch_size")
        kwargs.update({"learning_type": "onpolicy", "batch_size": self.minibatch_size})
        sign = [v.name for v in inspect.signature(super().__init__).parameters.values()]
        super().__init__(**{k: v for k, v in kwargs.items() if k in sign})
        self.train_collector: Collector
        self.test_collector: Collector
        self.logger: Logger
        self.log_dir = self.logger.writer.log_dir
        self.save_state = kwargs.get("save_state", False)
        self.state_map = kwargs.get("state_map", False)
        self.env_name = kwargs["task"]
        self.test_after_each_epoch = kwargs.get("test_after_each_epoch", False)
        self.state_freq = defaultdict(list)
        self.last_state_freq = defaultdict(list)
        self.success_rate = self.test_success_rate = 0.0
        self.extra_info = {}
        self.test_rew = 0.0
        self.cfg = kwargs.get("cfg")

    def __next__(self):
        result = super().__next__()
        self.log()
        return result

    def train_step(self):
        stop_fn_flag = False
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        result = self.train_collector.collect(n_step=self.step_per_collect)
        if result["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])
        self.logger.log_train_data(result, self.env_step)
        self.last_rew = result["rew"] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result["len"] if result["n/ep"] > 0 else self.last_len
        data = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result["n/ep"])),
            "n/st": str(int(result["n/st"])),
        }

        # useful info
        if result["n/ep"] > 0:
            self.success_rate = result["success_rate"]
            self.extra_info = result["extra_info"]
        return data, result, stop_fn_flag

    # call after train_step
    def policy_update_fn(self, data, result):
        self.policy.train()
        learn_info = self.policy.update(
            0, self.train_collector.buffer, minibatch_size=self.minibatch_size, repeat=self.repeat_per_collect
        )
        batch = learn_info.pop("batch")
        self.logger.log_info(batch, self.env_step)
        for k in [v + "_position" for v in ["x", "y", "z"]]:
            if k in batch.info:
                self.state_freq[k[0]].append(batch.info[k])
                self.last_state_freq[k[0]] = [batch.info[k]]
        self.train_collector.reset_buffer(keep_statistics=True)
        step = max([1] + [len(v) for v in learn_info.values() if isinstance(v, list)])
        self.gradient_step += step
        self.rew_coef = 0 if isinstance(learn_info["rew_coef"], list) else learn_info["rew_coef"]
        self.last_avg_ex_rew = learn_info["last_avg_ex_rew"]
        self.log_update_data(data, learn_info)

    def reset(self):
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0
        self.last_rew, self.last_len = 0.0, 0
        self.start_time = time.time()
        if self.train_collector is not None:
            self.train_collector.reset_stat()

        if self.test_collector is not None:
            self.test_collector.reset_stat()
            test_result = self.test_episode()
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = (test_result["rew"], test_result["rew_std"])
        if self.save_best_fn:
            self.save_best_fn(self.policy)

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0
        self.log()

    def test_step(self):
        stop_fn_flag = False
        test_stat = {}

        if self.test_after_each_epoch:
            test_result = self.test_episode()
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if self.best_epoch < 0 or self.best_reward < rew:
                self.best_epoch = self.epoch
                self.best_reward = float(rew)
                self.best_reward_std = rew_std
                if self.save_best_fn:
                    self.save_best_fn(self.policy)
            if not self.is_run:
                test_stat = {
                    "test_reward": rew,
                    "test_reward_std": rew_std,
                    "best_reward": self.best_reward,
                    "best_reward_std": self.best_reward_std,
                    "best_epoch": self.best_epoch,
                }
            if self.stop_fn and self.stop_fn(self.best_reward):
                stop_fn_flag = True
            self.test_rew = test_result["rew"]
            self.test_len = test_result["len"]
            self.test_success_rate = test_result["success_rate"]
            self.test_extra_info = test_result["extra_info"]
        return test_stat, stop_fn_flag

    def test_episode(self):
        self.policy.eval()
        self.test_collector.reset_env()
        self.test_collector.reset_buffer()
        result = self.test_collector.collect(n_episode=self.episode_per_test)
        self.logger.log_test_data(result, self.env_step)
        return result

    def log(self):
        @LogIt(self.log_dir + "/output.log")
        def save_info(*args, **kwargs):
            print(*args, **kwargs)

        if self.epoch == self.start_epoch:
            info = f"#{self.epoch:<2} ||| test | rew: {self.best_reward:8.2f} ± {self.best_reward_std:8.2f}"
        else:
            rtime = time_remain(time.time() - self.start_time, self.epoch, self.max_epoch, self.start_epoch)
            info = f"#{self.epoch:<2} train | rew: {self.last_rew:8.2f} | sr: {self.success_rate:4.0%} | len: {self.last_len:4.0f} | {rtime}"
            if self.verbose:
                info += f" | {self.extra_info} | rew_coef: {self.rew_coef:.2f}"
            if self.test_after_each_epoch:
                info += f" ||| test | rew: {self.test_rew:8.2f} | sr: {self.test_success_rate:4.0%} | len: {self.test_len:4.0f}"
                if self.verbose:
                    info += f" | {self.test_extra_info}"
        save_info(info)

    def log_update_data(self, data, losses):
        for k in losses.keys():
            self.stat[k].add(losses[k])
            losses[k] = self.stat[k].get()
            data[k] = f"{losses[k]:.3f}"
        self.logger.log_update_data(losses, self.env_step)


def onpolicy_trainer(*args, **kwargs):
    return OnpolicyTrainer(*args, **kwargs).run()
