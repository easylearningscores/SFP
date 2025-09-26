import torch
import torch.nn as nn
from backbone.simvp import SimVP
from backbone.convlstm import ConvLSTM
from backbone.earthformer import Earthformer
from backbone.tau import TAU 
from backbone.earthfarseer import EarthFarseer
from backbone.fno import FNO2d
from backbone.nmo import NMO 
from backbone.cno import CNO
from backbone.fourcastnet import FourCastNet 
from backbone.resnet_st import ResNet_ST 

class Agent(nn.Module):
    """
    The Agent for the SFP framework, also known as the Policy Network.

    The core function of this module is to:
    1. Receive the current environment state `s_t`.
    2. Process `s_t` using a powerful predictive model backbone.
    3. Output an "action" `a_t` whose dimensions match the latent space of the GWM.
    """
    def __init__(self, backbone_name, shape_in, latent_dim, out_channels, **backbone_kwargs):
        """
        Initializes the Agent.

        Args:
            backbone_name (str): The name of the backbone network to use.
            shape_in (tuple): The shape of the input data `s_t` (T_in, C_in, H, W).
            latent_dim (int): The hidden dimension within the backbone (a general parameter).
            out_channels (int): The number of output channels for the action a_t, which must match GWM's embedding_dim.
            **backbone_kwargs: A dictionary containing additional arguments required by the specific backbone.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        T_in, C_in, H, W = shape_in

        # --- 1. Initialize the Backbone ---
        if self.backbone_name.lower() == 'simvp-v2' or self.backbone_name.lower() == 'simvp':
            self.backbone = SimVP(shape_in=shape_in, hid_S=out_channels, hid_T=latent_dim, N_T=4, N_S=2, **backbone_kwargs)
        
        elif self.backbone_name.lower() == 'convlstm':
            self.backbone = ConvLSTM(input_channels=C_in, hidden_channels=[out_channels], kernel_size=[3], num_layers=1, **backbone_kwargs)
        
        elif self.backbone_name.lower() == 'earthformer':
            self.backbone = Earthformer(input_dim=C_in, output_dim=out_channels, **backbone_kwargs)
            
        elif self.backbone_name.lower() == 'tau':
            self.backbone = TAU(in_channels=C_in, out_channels=out_channels, **backbone_kwargs)
            
        elif self.backbone_name.lower() == 'earthfarseer':
            self.backbone = EarthFarseer(in_channels=C_in, out_channels=out_channels, **backbone_kwargs)
            
        elif self.backbone_name.lower() == 'fno':
            modes = backbone_kwargs.get('modes', 12)
            self.backbone = FNO2d(modes1=modes, modes2=modes, in_channels=T_in*C_in, out_channels=out_channels, **backbone_kwargs)
            self.reshape_for_fno = True
        
        elif self.backbone_name.lower() == 'nmo':
            self.backbone = NMO(in_channels=C_in, out_channels=out_channels, **backbone_kwargs)
            
        elif self.backbone_name.lower() == 'cno':
            self.backbone = CNO(in_channels=C_in, out_channels=out_channels, **backbone_kwargs)

        elif self.backbone_name.lower() == 'fourcastnet':
            self.backbone = FourCastNet(in_channels=C_in, out_channels=out_channels, **backbone_kwargs)

        elif self.backbone_name.lower() == 'resnet':
            self.backbone = ResNet_ST(in_channels=C_in, out_channels=out_channels, **backbone_kwargs)

        else:
            raise ValueError(f"Unknown backbone name: {self.backbone_name}")
        
        if not hasattr(self, 'reshape_for_fno'):
            self.reshape_for_fno = False

        # --- 2. Initialize the lightweight projection head (Projector) ---
        self.projector = nn.Conv2d(in_channels=out_channels, 
                                   out_channels=C_in, # The number of output channels should be the same as the input channels
                                   kernel_size=1)

    def get_action(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        Generates an action a_t based on the current state s_t.
        """
        B, T, C, H, W = s_t.shape
        
        if self.reshape_for_fno:
            s_t_reshaped = s_t.view(B, T * C, H, W)
            action = self.backbone(s_t_reshaped)
        else:
            action = self.backbone(s_t)

        # Ensure the action output has the shape [B, C_out, H', W']
        # Some models might output [B, T_out, C_out, ...], a single step needs to be taken
        if action.dim() == 5: # If the output is a sequence [B, T_out, C, H, W]
            action = action[:, -1, ...] # Take the last step of the sequence as the action

        return action

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        The complete forward pass, used for calculating the policy loss.
        Flow: State -> Action -> Projection back to the physical space.
        """
        action = self.get_action(s_t)
        projected_state = self.projector(action)
        
        return projected_state