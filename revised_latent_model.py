from torch import nn
import torch

from layers import ConvNorm, LinearNorm
from torch.nn import functional as F


class GMVAE(nn.Module):
    def __init__(self, hparams):
        super(GMVAE, self).__init__()
        self.latent_embedding_dim = hparams.latent_embedding_dim
        convolutions = []
        conv_layer_1 = nn.Sequential(
            ConvNorm(hparams.n_mel_channels,
                     hparams.latent_embedding_dim,
                     kernel_size=hparams.latent_kernel_size, stride=1,
                     padding=int((hparams.latent_kernel_size - 1) / 2),
                     dilation=1, w_init_gain='relu'),
            nn.BatchNorm1d(hparams.latent_embedding_dim))
        convolutions.append(conv_layer_1)

        conv_layer_2 = nn.Sequential(
            ConvNorm(hparams.latent_embedding_dim,
                     hparams.latent_embedding_dim,
                     kernel_size=hparams.latent_kernel_size, stride=1,
                     padding=int((hparams.latent_kernel_size - 1) / 2),
                     dilation=1, w_init_gain='relu'),
            nn.BatchNorm1d(hparams.latent_embedding_dim))
        convolutions.append(conv_layer_2)

        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.latent_embedding_dim,
                            int(hparams.latent_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

        self.linear_projection = LinearNorm(hparams.latent_embedding_dim, int(hparams.latent_embedding_dim / 2))

        self.linear_projection_mean = LinearNorm(int(hparams.latent_embedding_dim / 2), hparams.latent_out_dim)

        self.linear_projection_variance = LinearNorm(int(hparams.latent_embedding_dim / 2), hparams.latent_out_dim)

        self.fc3 = nn.Linear(hparams.latent_out_dim, int(hparams.latent_embedding_dim / 2))

        self.fc4 = nn.Linear(int(hparams.latent_embedding_dim / 2), hparams.latent_embedding_dim)

        self.happy_mean = nn.Parameter(torch.randn(torch.zeros((self.latent_output_dim)).view(self.latent_output_dim, 1).size()))

        self.happy_var = nn.Parameter(torch.randn(torch.zeros((self.latent_output_dim)).view(self.latent_output_dim, 1).size()))

        self.sad_mean = nn.Parameter(
            torch.randn(torch.zeros((self.latent_output_dim)).view(self.latent_output_dim, 1).size()))

        self.sad_var = nn.Parameter(
            torch.randn(torch.zeros((self.latent_output_dim)).view(self.latent_output_dim, 1).size()))

    def vae_encode(self, inputs):
        _, _, x, _, _ = inputs

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        out, _ = self.lstm(x)

        out = torch.mean(out, dim=1)
        x_after_mean = out

        out = self.linear_projection.forward(out)

        mean = self.linear_projection_mean.forward(out)

        variance = self.linear_projection_variance.forward(out)

        return mean, variance, x_after_mean

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, requires_grad=True)
        return mu + eps * std

    def decode(self, z, label=None):
        #  print('shape to be decoded', z.shape)
        # z=torch.cat([z, label],1)
        h3 = F.relu(self.fc3(z))
        # print('shape of the recons',h3.shape)
        #        pdb.set_trace()
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar, x_after_mean = self.vae_encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
