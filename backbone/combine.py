import torch
import torch.nn as nn
from multiconv import MultiConv
from fourier import FourierModule


class CombinedModel(nn.Module):
    def __init__(self, multi_conv_params, fourier_module_params):
        super(CombinedModel, self).__init__()
        self.multi_conv = MultiConv(**multi_conv_params)
        self.fourier_module = FourierModule(**fourier_module_params)

    def forward(self, x):
        output_multi_conv = self.multi_conv(x)
        output_fourier_module = self.fourier_module(x)
        output =  output_multi_conv +  output_fourier_module
        return output

if __name__ == "__main__":
    multi_conv_params = {
        'in_shape': (10, 7, 480, 480),
        'hid_S': 32,
        'hid_T': 128,
        'N_S': 2,
        'N_T': 4,
        'mlp_ratio': 8.,
        'drop': 0.0,
        'drop_path': 0.0,
        'spatio_kernel_enc': 3,
        'spatio_kernel_dec': 3,
        'pre_seq_length': 10,
        'aft_seq_length': 10,
    }

    fourier_module_params = {
        'img_size': [480, 480],
        'patch_size': 8,
        'in_channels': 7,
        'out_channels': 1,
        'embed_dim': 64,
        'input_frames': 10,
        'depth': 1,
        'mlp_ratio': 2.,
        'uniform_drop': False,
        'drop_rate': 0.,
        'drop_path_rate': 0.,
        'norm_layer': None,
        'dropcls': 0.,
    }

    combined_model = CombinedModel(multi_conv_params, fourier_module_params)
    input_frames = torch.rand(1, 10, 7, 480, 480)
    output_combined = combined_model(input_frames)
    print(output_combined.shape)
