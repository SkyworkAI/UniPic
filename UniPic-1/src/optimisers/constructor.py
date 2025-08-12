import inspect
import torch.nn as nn
from typing import List, Optional, Union
from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapper
from mmengine.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                               OPTIMIZERS)


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)

    num_decay_params = sum(p.numel() for p in decay)
    num_nodecay_params = sum(p.numel() for p in no_decay)
    print(f"num decayed parameter tensors: {len(decay)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(no_decay)}, with {num_nodecay_params:,} parameters")
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class MAROptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __call__(self, model: nn.Module) -> OptimWrapper:
        if hasattr(model, 'module'):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cls = self.optimizer_cfg['type']
        # Optimizer like HybridAdam in colossalai requires the argument name
        # `model_params` rather than `params`. Here we get the first argument
        # name and fill it with the model parameters.
        if isinstance(optimizer_cls, str):
            with OPTIMIZERS.switch_scope_and_registry(None) as registry:
                optimizer_cls = registry.get(self.optimizer_cfg['type'])
        fisrt_arg_name = next(
            iter(inspect.signature(optimizer_cls).parameters))
        # import pdb; pdb.set_trace()
        param_groups = add_weight_decay(model, optimizer_cfg.pop('weight_decay', 0))
        optimizer_cfg[fisrt_arg_name] = param_groups
        optimizer = OPTIMIZERS.build(optimizer_cfg)

        # # if no paramwise option is specified, just use the global setting
        # if not self.paramwise_cfg:
        #     optimizer_cfg[fisrt_arg_name] = model.parameters()
        #     optimizer = OPTIMIZERS.build(optimizer_cfg)
        # else:
        #     # set param-wise lr and weight decay recursively
        #     params: List = []
        #     self.add_params(params, model)
        #     optimizer_cfg[fisrt_arg_name] = params
        #     optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
