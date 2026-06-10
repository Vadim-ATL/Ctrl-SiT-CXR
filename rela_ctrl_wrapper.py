from timm.models.vision_transformer import PatchEmbed
import torch
import torch.nn as nn
from rela_ctrl_model import RelaCtrlBlock


class SiTRelaCtrlWrapper(nn.Module):
    """
    Wraps the base SiT model with the RelaCtrl framework.
    Only control_blocks and cond_embedder are trainable.
    Base SiT is fully frozen.
    """
    def __init__(self, base_sit_model, condition_channels=3,
                 relevant_layers=[2, 4, 6, 8, 10, 12, 14]):
        super().__init__()
        self.base_model      = base_sit_model
        self.relevant_layers = relevant_layers

        # Freeze base SiT entirely
        for p in self.base_model.parameters():
            p.requires_grad = False

        hidden_size  = self.base_model.x_embedder.proj.out_channels
        patch_size   = self.base_model.patch_size
        input_size   = self.base_model.x_embedder.img_size[0]

        # Trainable: condition patch embedder
        self.cond_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=condition_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        # Trainable: control blocks at selected layers only
        self.control_blocks = nn.ModuleDict({
            str(idx): RelaCtrlBlock(hidden_size) for idx in self.relevant_layers
        })

        self._init_weights()

    def _init_weights(self):
        w = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

    def forward(self, x, t, y, condition_img):
        """
        x:             (N, 4, H/8, W/8)  noisy latent
        t:             (N,)              timestep
        y:             (N,)              class label (int, 0=No_Finding 1=Pneumonia)
        condition_img: (N, 1, H/8, W/8) organ mask at latent resolution
        """
        x_main = self.base_model.x_embedder(x)             + self.base_model.pos_embed
        x_cond = self.cond_embedder(condition_img)          + self.base_model.pos_embed

        t_emb  = self.base_model.t_embedder(t)
        y_emb  = self.base_model.y_embedder(y, self.training)
        c      = t_emb + y_emb

        # Control stream initialised as sum of main + condition embeddings
        x_ctrl = x_main + x_cond

        for i, block in enumerate(self.base_model.blocks):
            x_main = block(x_main, c)

            if i in self.relevant_layers:
                ctrl_block              = self.control_blocks[str(i)]
                main_injection, x_ctrl  = ctrl_block(x_ctrl, c)
                x_main                  = x_main + main_injection

        x_main = self.base_model.final_layer(x_main, c)
        x_out  = self.base_model.unpatchify(x_main)

        # FIX: do NOT chunk — return the full 4-channel velocity prediction
        # Original code did x_out.chunk(2, dim=1) which discarded half the output
        return x_out

    def forward_with_cfg(self, x, t, y, cfg_scale, condition_img, **kwargs):
        """
        Classifier-Free Guidance forward.
        Expects x to be the doubled batch [cond; uncond] assembled by the caller.
        """
        model_out = self.forward(x, t, y, condition_img=condition_img, **kwargs)

        # Split cond / uncond halves
        cond_out, uncond_out = model_out.chunk(2, dim=0)

        # CFG extrapolation
        guided = uncond_out + cfg_scale * (cond_out - uncond_out)

        # Return doubled batch so ODE sampler can chunk again and take first half
        return torch.cat([guided, guided], dim=0)