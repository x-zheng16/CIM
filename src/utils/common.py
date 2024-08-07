import csv
import datetime
import os
import random
import re
from collections import defaultdict
from distutils.util import strtobool
from functools import wraps

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical, Independent, Normal

auto_bool = lambda x: bool(strtobool(x) if isinstance(x, str) else x)


# log
def time_spent(time):
    output = str(datetime.timedelta(seconds=int(time)))
    return output


def time_remain(time, epoch, nepoch, last_epoch=0):
    time = time / (epoch - last_epoch) * (nepoch - epoch)
    output = "remain " + str(datetime.timedelta(seconds=int(time)))
    return output


class LogIt:
    def __init__(self, logfile="out.log"):
        self.logfile = logfile

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            with open(self.logfile, mode="a", encoding="utf-8") as opened_file:
                output = list(map(str, args)) if len(args) else []
                output += [f"{k}={v}" for k, v in kwargs.items()]
                opened_file.write(", ".join(output) + "\n")
            return func(*args, **kwargs)

        return wrapped_function


# net
def grad_monitor(net):
    grad_norm = 0
    for name, param in net.named_parameters():
        if param.grad is not None:
            if torch.all(~torch.isnan(param.grad)):
                grad_norm += torch.linalg.vector_norm(param.grad.detach()).item() ** 2
            else:
                print(f"grad of param {name} is nan")
    return np.sqrt(grad_norm)


def param_monitor(net):
    param_norm = []
    for m in net.children():
        if hasattr(m, "weight"):
            param_norm.append(torch.linalg.matrix_norm(m.weight.detach(), 2).item())
    return np.mean(param_norm) if len(param_norm) else 0


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def last_layer_init(net):
    m = net.mean.mlp[-1]
    m.weight.data.copy_(0.01 * m.weight.data)


def sync_weight(target_net, online_net, tau):
    for o, n in zip(target_net.parameters(), online_net.parameters()):
        o.data.copy_(o.data * tau + n.data * (1 - tau))


def disable_grad(model):
    for p in model.parameters():
        p.requires_grad = False


def enable_grad(model):
    for p in model.parameters():
        p.requires_grad = True


# model
def set_seed(seed):
    # pytorch
    torch.manual_seed(seed)

    # according to https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)


def get_env_space(env, index=None):
    obs_space = env.observation_space
    act_space = env.action_space
    if isinstance(obs_space, list):
        obs_space = obs_space[0]
        act_space = act_space[0]
    if hasattr(obs_space, "spaces"):
        if isinstance(obs_space.spaces, list):
            obs_space = obs_space.spaces[index or 0]
        # if isinstance(obs_space.spaces, dict):
        #     obs_space = obs_space["observation"]
    if hasattr(act_space, "spaces"):
        if isinstance(act_space.spaces, list):
            act_space = act_space.spaces[index or 0]
    return obs_space, act_space


def get_dist_fn(act_space):
    if isinstance(act_space, Discrete):
        dist_fn = lambda logits: Categorical(logits=logits)
    elif isinstance(act_space, Box):
        dist_fn = lambda *logits: Independent(Normal(*logits), 1)
    return dist_fn


def split_batch(minibatch_size, batch_size, shuffle=False):
    indices = np.random.permutation(batch_size) if shuffle else np.arange(batch_size)
    for idx in range(0, batch_size, minibatch_size):
        if idx + minibatch_size * 2 >= batch_size:
            yield indices[idx:]
            break
        yield indices[idx : idx + minibatch_size]


# files
def find_all_files(
    root_dir,
    pattern,
    suffix=None,
    prefix=None,
    return_pattern=False,
    exclude_suffix=(".png", ".txt", ".log", "config.json", ".pdf", ".yml"),
):
    file_list = []
    pattern_list = []
    if os.path.isfile(root_dir):
        m = re.search(pattern, root_dir)
        if m is not None:
            file_list.append(root_dir)
            pattern_list.append(m.groups())
    else:
        for dirname, _, files in os.walk(root_dir):
            for f in files:
                if suffix and not f.endswith(suffix):
                    continue
                elif f.endswith(exclude_suffix):
                    continue
                elif prefix and not f.startswith(prefix):
                    continue
                absolute_path = os.path.join(dirname, f)
                m = re.search(pattern, absolute_path)
                if m is not None:
                    file_list.append(absolute_path)
                    pattern_list.append(m.groups())
    if return_pattern:
        return file_list, pattern_list
    else:
        return file_list


def group_files(file_list, pattern):
    res = defaultdict(list)
    for f in file_list:
        m = re.search(pattern, f) if pattern else None
        res[m.group(1) if m else ""].append(f)
    return res


def csv2numpy(csv_file):
    csv_dict = defaultdict(list)
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                csv_dict[k].append(eval(v))
    return {k: np.array(v) for k, v in csv_dict.items()}


# stat
class RunningMeanStd(object):
    def __init__(self, mean=0.0, var=1.0, clip_max=10.0, epsilon=1e-8, device="cuda"):
        self.mean, self.var = mean, var
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon
        self.device = device

    def norm(self, data_array):
        data_array = (data_array - self.mean) / torch.sqrt(self.var + self.eps)
        if self.clip_max:
            data_array = torch.clamp(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array):
        batch_mean, batch_var = torch.mean(data_array, 0), torch.var(data_array, 0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

    def load(self, rms):
        self.mean = self._to_torch(rms.mean)
        self.var = self._to_torch(rms.var)
        self.clip_max = rms.clip_max
        self.count = rms.count
        self.eps = rms.eps

    def _to_torch(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)


# estimator
from pykeops.torch import Vi, Vj

pbe_fn = lambda x, y, k: ((Vi(x) - Vj(y)) ** 2).sum().Kmin(k, 1)


# Particle-Based Estimator (PBE)
class PBE:
    def __init__(self, k, buffer_size, style):
        super().__init__()
        self.k = k
        self.buffer_size = buffer_size
        self.style = style
        self.ptr = self.max_ptr = 0
        self.buf = None

    def get_rew(self, x, y):
        if len(x) == 1:
            rew = torch.zeros(1)
        else:
            k = min(len(x) - 1, self.k)
            r = pbe_fn(x, y, k + 1)[:, 1:].sqrt()
            if self.style == "log_mean":
                rew = (1 + r.mean(-1)).log()
            elif self.style == "mean":
                rew = r.mean(-1)
            elif self.style == "log":
                rew = (1 + r[:, -1]).log()
            else:
                raise Exception(f"unsupported PBE style")
        return rew

    def get_xy(self, x, use_buffer=False):
        if use_buffer:
            self.update_buffer(x)
            return x, self.buf[: self.max_ptr]
        else:
            return x, x

    def update_buffer(self, x):
        B = len(x)
        if self.buf is None:
            self.buf = torch.zeros(self.buffer_size, *x.shape[1:], device=x.device)
        if self.ptr + B > self.buffer_size:
            sub = self.buffer_size - self.ptr
            self.buf[self.ptr :] = x[:sub]
            self.buf[: B - sub] = x[sub:]
            self.ptr = B - sub
            self.max_ptr = self.buffer_size
        else:
            self.buf[self.ptr : self.ptr + B] = x
            self.ptr += B
        self.max_ptr = max(self.ptr, self.max_ptr)


# operator
def masked_logsumexp(input, dim, keepdim=False, mask=None):
    mask = mask.bool()
    max_offset = -1e7 * mask if mask is not None else 0
    s = (input + max_offset).max(dim, True)[0]
    input_offset = input - s
    if mask is not None:
        input_offset.masked_fill_(mask, -float("inf"))
    outputs = s + input_offset.exp().sum(dim, True).log()
    if not keepdim:
        outputs = outputs.squeeze(-1)
    return outputs
