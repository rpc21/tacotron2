from torch import nn
import torch
from torch.distributions import Normal

from layers import ConvNorm, LinearNorm
from torch.nn import functional as F
from utils import to_gpu
import pdb


class GMVAE_revised(nn.Module):
    def __init__(self, hparams, supervised=False):
        super(GMVAE_revised, self).__init__()
        self.latent_embedding_dim = hparams.latent_embedding_dim
        self.supervised = supervised
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

        # self.mean_pool = nn.AvgPool1d(hparams.latent_kernel_size, stride=1)
        #
        # self.mean_pool_out_size = hparams.latent_embedding_dim - hparams.latent_kernel_size + 1

        self.linear_projection = LinearNorm(hparams.latent_embedding_dim, int(hparams.latent_embedding_dim / 2))

        self.linear_projection_mean = LinearNorm(int(hparams.latent_embedding_dim / 2), hparams.latent_out_dim)

        self.linear_projection_variance = LinearNorm(int(hparams.latent_embedding_dim / 2), hparams.latent_out_dim)

        self.fc3 = nn.Linear(hparams.latent_out_dim, int(hparams.latent_embedding_dim / 2))

        self.fc4 = nn.Linear(int(hparams.latent_embedding_dim / 2), hparams.latent_embedding_dim)


    def parse_batch(self, batch):
        if self.supervised:
            text_padded, input_lengths, mel_padded, gate_padded, output_lengths, mel_padded_512, gate_padded_512, output_lengths_512, labels = batch
        else:
            text_padded, input_lengths, mel_padded, gate_padded, output_lengths, mel_padded_512, gate_padded_512, output_lengths_512 = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        mel_padded_512 = to_gpu(mel_padded_512).float()
        gate_padded_512 = to_gpu(gate_padded_512).float()
        output_lengths_512 = to_gpu(output_lengths_512).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, mel_padded),
            (mel_padded, gate_padded))


    def vae_encode(self, inputs):
        _, _, x, _, _, _ = inputs
#        print('x shape:', x.shape)
#        pdb.set_trace()
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
#            pdb.set_trace()
        x = x.transpose(1, 2)
#        print('Just finished convs')
#        pdb.set_trace()
        out, _ = self.lstm(x)
#        print('Just finished lstm', out.shape)
#        pdb.set_trace()
        out = torch.mean(out, dim=1)
        x_after_mean = out
#        print('After mean pool', out.shape)
#        pdb.set_trace()
        out = self.linear_projection.forward(out)
#        print('After linear 1', out.shape)
#        pdb.set_trace()
        mean = self.linear_projection_mean.forward(out)
        variance = self.linear_projection_variance.forward(out)
        #        mean = torch.mean(torch.mean(self.linear_projection_mean.forward(out),dim=1), dim=0)
        #        variance = torch.mean(torch.mean(self.linear_projection_variance.forward(out),dim=1), dim=0)
        #    print('mean', mean.shape)
        #   print('variance', variance.shape)
#        pdb.set_trace()
        return mean, variance, x_after_mean


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        #  print('shape to be decoded', z.shape)
        h3 = F.relu(self.fc3(z))
        # print('shape of the recons',h3.shape)
        #        pdb.set_trace()
        return torch.sigmoid(self.fc4(h3))


    def forward(self, x):
        mu, logvar, x_after_mean = self.vae_encode(x)
        z = self.reparameterize(mu, logvar)
        # print('mu shape:', mu.shape)
        # print('logvar shape:', logvar.shape)
        #       pdb.set_trace()
        return self.decode(z), mu, logvar, x_after_mean

    
    def generate_sample(self, x):
        mu, logvar, _ = self.vae_encode(x)
        #        pdb.set_trace()
#        pdb.set_trace()
        return Normal(mu, logvar.exp()).sample((1,x[2].shape[2])).squeeze(dim=0)
