# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import basis_rotation
import torch
import torch.optim
import torch.nn as nn
import time
import os
import math
from collections import deque
import torch.nn.functional as F
import logging
from typing import cast
from collections import OrderedDict

import gc

class Version:
    def __init__(self, version=0):
        self.version = version

    def __repr__(self):
        return "v%d" % self.version

    def incr(self):
        return Version(version=self.version+1)

class MultipleOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        
        # Initialize empty hooks to satisfy PyTorch 2.0+ internals/Dynamo
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
        self.defaults = {}

    def zero_grad(self, set_to_none: bool = False):
        for op in self.optimizers:
            op.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for op in self.optimizers:
            op.step()
        return loss

    @property
    def param_groups(self):
        return [p for op in self.optimizers for p in op.param_groups]

    @property
    def state(self):
        return {p: s for op in self.optimizers for p, s in op.state.items()}

    def state_dict(self):
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        if not isinstance(state_dicts, list) or len(state_dicts) != len(self.optimizers):
            raise ValueError("MultipleOptimizer load_state_dict expects a list matching num optimizers")
        for op, sd in zip(self.optimizers, state_dicts):
            op.load_state_dict(sd)
            
    def __getattr__(self, key):
        return getattr(self.optimizers[0], key)

class OptimizerWithWeightStashing(torch.optim.Optimizer):
    """Wrapper class that adds weight stashing to a vanilla torch.optim.Optimizer.

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding
                      base_optimizer (torch.optim.{optim_name}).
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """

    def __init__(self, optim_name, modules, master_parameters,
                 num_versions, verbose_freq=0, macrobatch=False,
                 clip_grad=None, save_dir=None, stash_to_cpu=False,
                 accumulation_steps=1, **optimizer_args):
        self.modules = modules
        self.master_parameters = master_parameters
        self.clip_grad = clip_grad
        self.save_dir = save_dir
        self.stash_to_cpu = stash_to_cpu
        self.optim_name = optim_name
        # Only need at most 2 versions if using macrobatching.
        if macrobatch:
            num_versions = min(2, num_versions)
        num_versions = num_versions // accumulation_steps
        self.num_versions = num_versions
        self.accmulation_steps = accumulation_steps
        self.base_optimizer = None
        if len(self.master_parameters) == 0:
            logging.info("Warning: no parameter groups to optimize")
        else:
            max_precond_dim = optimizer_args.get('max_preconditioner_dim', 10000)

            params_rotation = []
            params_nonrotation = []
            target_modules_list = ["attn", "mlp", "sa", "ffwd"]
            for module in self.modules:
                for name, mod in module.named_modules():
                    if not isinstance(mod, nn.Linear):
                        continue
                    if not any(target_key in name for target_key in target_modules_list):
                        continue
                    params_rotation.append(mod.weight)
            id_params_rotation = [id(p) for p in params_rotation]
            params_nonrotation = [p for module in self.modules for p in module.parameters() if id(p) not in id_params_rotation]

            if optim_name == 'BasisRotation':
                optimizers = []
                if len(params_nonrotation) > 0:
                    adamw_optimizer = torch.optim.AdamW(
                        params_nonrotation,
                        lr=optimizer_args['lr'],
                        betas=optimizer_args.get('betas', (0.9, 0.999)),
                        weight_decay=optimizer_args.get('weight_decay', 0),
                    )
                    optimizers.append(adamw_optimizer)
                if len(params_rotation) > 0:
                    rotatedadam_optimizer = basis_rotation.BasisRotation(params_rotation, **optimizer_args)
                    optimizers.append(rotatedadam_optimizer)
                self.base_optimizer = MultipleOptimizer(optimizers)
            else:
                self.base_optimizer = getattr(torch.optim, optim_name)(
                        master_parameters, **optimizer_args)
        self.latest_version = Version()
        self.current_version = Version()
        self.initialize_queue()
        self.verbose_freq = verbose_freq
        self.batch_counter = 0
        
        # If macrobatching, push and pop versions at the right rate.
        if macrobatch:
            self.update_interval = self.num_versions
        else:
            self.update_interval = accumulation_steps

    def __getattr__(self, key):
        """Relay the unknown key to base_optimizer."""
        if self.base_optimizer is None: # handle empty parameter list case
            if key == "state":
                return {}
            if key == "param_groups":
                return [{'params': []}]
            else:
                return None
        return getattr(self.base_optimizer, key)
    
    def append_to_queue(self, data):
        state_dicts, version = data
        if self.save_dir is not None:
            # only keep the filename in memory and load it when needed
            fname = os.path.join(self.save_dir, f"version_{version.version % self.num_versions}.pth.tar")
            d = {"state_dicts": state_dicts, "version": version}
            torch.save(d, fname)
            self.queue.append(fname)
        else:
            self.queue.append((state_dicts, version))

    def get_from_queue(self, index):
        if self.save_dir is not None:
            fname = self.queue[index]
            d = torch.load(fname)
            return d["state_dicts"], d["version"]
        else:
            return self.queue[index]

    def insert_to_queue(self, data, index):
        state_dicts, version = data
        if self.save_dir is not None:
            fname = os.path.join(self.save_dir, f"version_{index}.pth.tar")
            d = {"state_dicts": state_dicts, "version": version}
            torch.save(d, fname)
            self.queue[index] = (fname)
        else:
            self.queue[index] = ((state_dicts, version))

    def initialize_queue(self):

        # 1. Kill the reference to the old data
        self.queue = None
        self.buffered_state_dicts = None
        
        # 2. NOW collect garbage (Python sees the old data is unreferenced)
        gc.collect()
        
        # 3. Clear PyTorch's internal cache to ensure the VRAM is actually available
        torch.cuda.empty_cache()

        # 4. Allocate the new queue
        self.queue = deque(maxlen=self.num_versions)
        for i in range(self.num_versions):
            self.append_to_queue(self.get_params(clone=True))
        self.buffered_state_dicts = self.get_from_queue(0)[0]

    def get_params(self, clone):
        if clone:
            state_dicts = []
            for module in self.modules:
                state_dict = module.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].clone().cpu() if self.stash_to_cpu else state_dict[key].clone()
                state_dicts.append(state_dict)
        else:
            for i, module in enumerate(self.modules):
                state_dict = module.state_dict()
                for key in state_dict:
                    if "running_" in key:
                        continue
                    if "mask" in key:
                        self.buffered_state_dicts[i][key] = state_dict[key].clone().cpu() if self.stash_to_cpu else state_dict[key].clone()
                    else:
                        self.buffered_state_dicts[i][key].copy_(state_dict[key].cpu() if self.stash_to_cpu else state_dict[key])
            state_dicts = self.buffered_state_dicts
        return state_dicts, self.latest_version

    def set_params(self, state_dicts, version):
        for (state_dict, module) in zip(state_dicts, self.modules):
            cur_state_dict = module.state_dict()
            for key in state_dict:
                # running_mean/var should accumulate normally; mask shapes may differ
                if "running_" in key or "mask" in key:
                    state_dict[key] = cur_state_dict[key].cuda() if self.stash_to_cpu else cur_state_dict[key]
            module.load_state_dict(state_dict)

            for key in state_dict:
                if "mask" in key:
                    attribute_names = key.split(".")
                    attribute = module
                    for attribute_name in attribute_names:
                        attribute = getattr(attribute, attribute_name)
                    attribute = state_dict[key].cuda() if self.stash_to_cpu else state_dict[key]
        self.current_version = version

    def load_old_params(self):
        if self.num_versions > 1:
            self.set_params(*self.get_from_queue(0))

    def load_new_params(self):
        if self.num_versions > 1:
            self.set_params(*self.get_from_queue(-1))

    def zero_grad(self):
        if self.base_optimizer is not None and self.batch_counter % self.update_interval == 0:
            self.base_optimizer.zero_grad()
    
    def train(self):
        self.base_optimizer.train()

    def eval(self):
        self.base_optimizer.eval()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        # Update the gradient every `update_interval` steps.
        if self.batch_counter % self.update_interval != self.update_interval - 1:
            self.batch_counter += 1
            return None

        log_timing = self.verbose_freq > 0 and self.batch_counter % self.verbose_freq == 0
        if log_timing:
            start_time = time.time()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.div_(self.update_interval)

        if self.clip_grad is not None:
            for group in self.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.clip_grad)
            
        loss = self.base_optimizer.step() if self.base_optimizer is not None else None
        self.latest_version = self.latest_version.incr()
        if self.num_versions > 1:
            self.buffered_state_dicts = self.get_from_queue(0)[0]
            self.append_to_queue(self.get_params(clone=False))

        if log_timing:
            logging.info("Optimizer step took: %.3f" % (time.time() - start_time))
        self.batch_counter += 1

        return loss