import torch
import torch.nn as nn

class ZeroLinear(nn.Module):
    """
    Zero-initialized linear layer to ensure the control branch 
    does not disrupt the pre-trained base model at the start of training.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)

class TDSM_Placeholder(nn.Module):
    """
    Two-Dimensional Shuffle Mixer (TDSM).
    In the RelaCtrl paper, this replaces the heavy Attention + MLP of standard ControlNet.
    It groups tokens and channels to perform local mixing. 
    
    *Note: This is a structurally representative lightweight mixer. For exact reproduction, 
    you would implement the precise random token/channel shuffling logic detailed in the paper.*
    """
    def __init__(self, hidden_size, reduction=4):
        super().__init__()
        mid_size = hidden_size // reduction
        self.mixer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid_size),
            nn.SiLU(),
            nn.Linear(mid_size, hidden_size)
        )
        
    def forward(self, x):
        return self.mixer(x)

class RelaCtrlBlock(nn.Module):
    """
    Relevance-Guided Lightweight Control Block (RGLC).
    Takes the control signal, modulates it, applies TDSM, and bridges it to the main branch.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.tdsm = TDSM_Placeholder(hidden_size)
        self.zero_linear = ZeroLinear(hidden_size)
        
        # Adaptive Layer Norm to condition the control block on timestep/class labels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x_ctrl, c):
        # 1. Modulate the control signal with the combined timestep/label embeddings
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_ctrl_mod = x_ctrl * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # 2. Mix features efficiently
        x_ctrl_mixed = self.tdsm(x_ctrl_mod)
        
        # 3. Create the zero-initialized projection for the main branch
        main_injection = self.zero_linear(x_ctrl_mixed)
        
        # Return the injection payload, and the updated control stream for the next block
        return main_injection, x_ctrl + x_ctrl_mixed
    
    