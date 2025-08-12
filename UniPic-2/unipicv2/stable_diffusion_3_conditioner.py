import torch
import torch.nn as nn
# from transformers.modeling_utils import PreTrainedModel
from diffusers.configuration_utils import register_to_config, ConfigMixin
from unipicv2.modeling_connector import ConnectorEncoder
from unipicv2.configuration_connector import ConnectorConfig
from diffusers.models.modeling_utils import ModelMixin


class StableDiffusion3Conditioner(ModelMixin, ConfigMixin):
    model_type: str = "sd3_conditioner"  # stored into config for hub niceties

    @register_to_config
    def __init__(
        self,
        connector_config: dict,                 # dict passed to ConnectorConfig(**connector)
        num_queries: int = 256,
        llm_hidden_size: int = 3584,
        pooled_projection_dim: int = 2048,
        joint_attention_dim: int = 4096,
    ):
        super().__init__()

        self.connector = ConnectorEncoder(ConnectorConfig(**connector_config))
        self.projector_1 = nn.Linear(llm_hidden_size, self.connector.config.hidden_size)
        self.projector_2 = nn.Linear(self.connector.config.hidden_size, pooled_projection_dim)
        self.projector_3 = nn.Linear(self.connector.config.hidden_size, joint_attention_dim)
        self.meta_queries = nn.Parameter(torch.zeros(num_queries, llm_hidden_size))

    def _init_weights(self, module):
        pass

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, llm_hidden_size)
        Returns:
          prompt_embeds: (batch, seq_len, joint_attention_dim)
          pooled_prompt_embeds: (batch, pooled_projection_dim)
        """
        x = self.projector_1(x)
        x = self.connector(x)  # expects (B, L, hidden)
        pooled_prompt_embeds = self.projector_2(x.mean(1))
        prompt_embeds = self.projector_3(x)

        return prompt_embeds, pooled_prompt_embeds



if __name__ == "__main__":
    import torch
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    pretrained_model_name_or_path = "stabilityai/stable-diffusion-3.5-medium"

    conditioner = StableDiffusion3Conditioner(
        num_queries=256,
        connector_config=dict(
            hidden_size=1536,
            intermediate_size=8960,
            num_hidden_layers=24,
            _attn_implementation='flash_attention_2',
            num_attention_heads=24, ),
        llm_hidden_size=3584,
        pooled_projection_dim=2048,
        joint_attention_dim=4096,
    ).bfloat16()

    checkpoint = torch.load(args.checkpoint)

    info = conditioner.load_state_dict(checkpoint, strict=False)
    import pdb; pdb.set_trace()

    os.makedirs(args.output, exist_ok=True)

    conditioner.save_pretrained(args.output)
