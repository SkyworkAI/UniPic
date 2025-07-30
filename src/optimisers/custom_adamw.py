import inspect
from torch.optim import AdamW


class CustomAdamW(AdamW):
    def __init__(self, params, weight_decay, *args, **kwargs):
        import pdb; pdb.set_trace()
        if isinstance(params, dict):
            params = [p for p in params.values() if p.requires_grad]
        else:
            params = [p for p in params if p.requires_grad]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(AdamW).parameters
        # extra_args = dict(fused=True) if fused_available else dict()
        # print(f"using fused AdamW: {fused_available}")

        # kwargs.update(extra_args)

        super().__init__(params=optim_groups, *args, **kwargs)
