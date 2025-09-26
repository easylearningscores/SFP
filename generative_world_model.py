import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        transpose=False,
        act_norm=False
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvDynamicsLayer(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvDynamicsLayer, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class AtmosphericEncoder(nn.Module):
    def __init__(self, C_in, spatial_hidden_dim, num_spatial_layers):
        super(AtmosphericEncoder, self).__init__()
        strides = stride_generator(num_spatial_layers)
        self.enc = nn.Sequential(
            ConvDynamicsLayer(C_in, spatial_hidden_dim, stride=strides[0]),
            *[ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=s) for s in strides[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class AtmosphericDecoder(nn.Module):
    def __init__(self, spatial_hidden_dim, C_out, num_spatial_layers):
        super(AtmosphericDecoder, self).__init__()
        strides = stride_generator(num_spatial_layers, reverse=True)
        self.dec = nn.Sequential(
            *[ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=s, transpose=True) for s in strides[:-1]],
            # Change the input channels to match concatenated hid (spatial_hidden_dim) + enc1 (spatial_hidden_dim)
            ConvDynamicsLayer(2 * spatial_hidden_dim, spatial_hidden_dim, stride=strides[-1], transpose=True)
        )
        # Add a projection layer to match enc1 dimensions if needed
        self.proj = nn.Conv2d(spatial_hidden_dim, spatial_hidden_dim, 1) if spatial_hidden_dim != C_out else nn.Identity()
        self.readout = nn.Conv2d(spatial_hidden_dim, C_out, 1)

    def forward(self, hid, enc1=None):
        # Project enc1 to match hid dimensions if needed
        if enc1 is not None:
            enc1 = self.proj(enc1)
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        if enc1 is not None:
            Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        else:
            Y = self.dec[-1](hid)
        Y = self.readout(Y)
        return Y
        
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5, top_k=3):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._top_k = top_k

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        top_k_indices = torch.topk(distances, self._top_k, dim=1, largest=False)[1]
        top_k_encodings = []
        top_k_quantized = []
        for i in range(self._top_k):
            encoding_indices = top_k_indices[:, i].unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            top_k_encodings.append(encodings)

            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
            top_k_quantized.append(quantized)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(top_k_quantized[0].detach(), inputs)  
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (top_k_quantized[0] - inputs).detach()  
        top_k_quantized = [inputs + (q - inputs).detach() for q in top_k_quantized] 

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, top_k_quantized

    def lookup(self, x):
        embeddings = F.embedding(x, self._embedding)
        return embeddings

class top_k_quantization(nn.Module):
    def __init__(self,
                 in_channel=1,
                 num_hiddens=128,
                 res_layers=2,
                 # res_units is removed
                 embedding_nums=1024,  # K
                 embedding_dim=128,  # D
                 top_k=1,
                 commitment_cost=0.25,
                 decay=0.99):
        super(top_k_quantization, self).__init__()
        self.in_channel = in_channel
        self.num_hiddens = num_hiddens
        self.res_layers = res_layers
        # self.res_units is removed
        self.embedding_dim = embedding_dim
        self.embedding_nums = embedding_nums
        self.top_k = top_k
        self.decay = decay
        self.commitment_cost = commitment_cost
        
        # Replace old encoder with AtmosphericEncoder
        self._encoder = AtmosphericEncoder(in_channel, num_hiddens, self.res_layers)
        
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        
        # code book
        self._vq_vae = VectorQuantizerEMA(num_embeddings=self.embedding_nums,
                                  embedding_dim=self.embedding_dim,
                                  commitment_cost=self.commitment_cost,
                                  decay=self.decay,
                                  top_k=self.top_k)

        # Replace old decoder with AtmosphericDecoder
        self._decoder = AtmosphericDecoder(embedding_dim, in_channel, res_layers)

    def forward(self, x):
        '''
        Process input x through the improved neural network model.
        The shape of the input is [B, C, W, H], i.e. [batch size, number of channels, width, height].
        The input x is encoded by the AtmosphericEncoder which outputs latent features and skip connections.
        Convolution is then performed before applying VQ-VAE, and z is transformed into coded form.
        After VQ-VAE processing, the output consists of the loss, quantized encoding, perplexity, etc.
        The quantized code is reconstructed by the AtmosphericDecoder which uses skip connections for better reconstruction.
        '''
        latent, skip = self._encoder(x)
        z = self._pre_vq_conv(latent)
        loss, quantized, perplexity, _, quantized_list = self._vq_vae(z)
        x_recon = self._decoder(quantized, skip)
        return loss, x_recon, perplexity

    def get_embedding(self, x):
        latent, _ = self._encoder(x)
        return self._pre_vq_conv(latent)

    def get_quantization(self, x):
        latent, _ = self._encoder(x)
        z = self._pre_vq_conv(latent)
        _, quantized, _, _, _ = self._vq_vae(z)
        return quantized

    def reconstruct_img_by_embedding(self, embedding):
        loss, quantized, perplexity, _, quantized_list = self._vq_vae(embedding)
        # Note: Skip connection not available in this path
        return self._decoder(quantized)

    def reconstruct_img(self, q):
        # Note: Skip connection not available in this path
        return self._decoder(q)

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

class Generative_World_Model(nn.Module):
    def __init__(self,
                 in_channel=1,
                 res_layers=1,
                 embedding_nums=1024, 
                 embedding_dim=256,
                 top_k=3):
        super(Generative_World_Model, self).__init__()
        self.BV_module = top_k_quantization(
            in_channel=in_channel,
            num_hiddens=embedding_dim, 
            res_layers=res_layers,
            embedding_dim=embedding_dim,
            embedding_nums=embedding_nums,
            top_k=top_k
        )
        
    def forward(self, input_frames):
        latent, skip = self.BV_module._encoder(input_frames)
        z = self.BV_module._pre_vq_conv(latent)
        vq_loss, Latent_embed, _, _, Latent_embed_list = self.BV_module._vq_vae(z)
        pred = self.BV_module._decoder(Latent_embed, skip)

        top_k_features = []
        for quantized_top_k in Latent_embed_list:
            quantized_top_k = quantized_top_k.permute(0,3,1,2)
            predicti_feature_k = self.BV_module._decoder(quantized_top_k, skip)
            top_k_features.append(predicti_feature_k)
        return pred, top_k_features, vq_loss

if __name__ == '__main__':
    # --- Test Run ---
    input_tensor = torch.rand(1, 69, 180, 360)
    model = Generative_World_Model(in_channel=69,
                     res_layers=2,
                     embedding_nums=1024, 
                     embedding_dim=256,
                     top_k=10)
    
    pred, top_k_features, vq_loss = model(input_tensor)
    print(f"Prediction shape: {pred.shape}")
    print(f"Number of top_k features: {len(top_k_features)}")
    print(f"First top_k feature shape: {top_k_features[0].shape}")
    print(f"VQ-Loss: {vq_loss}")