from torch import nn
import numpy as np
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
        self.num_lables = hparams.num_lables
        self.k = hparams.num_of_mixtues
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

        self.linear_projection = LinearNorm(hparams.latent_embedding_dim + self.num_lables,
                                            int(hparams.latent_embedding_dim / 2))

        self.linear_projection_mean_variance = LinearNorm(int(hparams.latent_embedding_dim / 2),
                                                          hparams.latent_out_dim * self.k * 2)

        # self.linear_projection_variance = LinearNorm(int(hparams.latent_embedding_dim / 2), hparams.latent_out_dim)

        self.fc3 = nn.Linear(hparams.latent_out_dim*self.k + self.num_lables, int(hparams.latent_embedding_dim / 2))

        self.fc4 = nn.Linear(int(hparams.latent_embedding_dim / 2), hparams.latent_embedding_dim)

        self.z_init = torch.nn.Parameter(
            torch.randn(1, 2 * self.k, self.latent_embedding_dim) / np.sqrt(self.k * self.latent_embedding_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter((torch.ones(self.k) / self.k), requires_grad=False)

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

    def vae_encode(self, inputs, label=None):
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
#        out = torch.cat([out, label], 1)
        out = self.linear_projection.forward(out)
        #        print('After linear 1', out.shape)
        #        pdb.set_trace()
        mean_variance = self.linear_projection_mean_variance.forward(out)
        # variance = self.linear_projection_variance.forward(out)
        #        mean = torch.mean(torch.mean(self.linear_projection_mean.forward(out),dim=1), dim=0)
        #        variance = torch.mean(torch.mean(self.linear_projection_variance.forward(out),dim=1), dim=0)
        #    print('mean', mean.shape)
        #   print('variance', variance.shape)
        #        pdb.set_trace()
        mean, variance = self.gaussian_parameters(mean_variance)
        return mean, variance, x_after_mean

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, label=None):
        #  print('shape to be decoded', z.shape)
#        z = torch.cat([z, label], 1)
        pdb.set_trace()
        h3 = F.relu(self.fc3(z))
        # print('shape of the recons',h3.shape)
        #        pdb.set_trace()
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, label=None):
        mu, logvar, x_after_mean = self.vae_encode(x, label)
        z = self.reparameterize(mu, logvar)
        pdb.set_trace()
        # print('mu shape:', mu.shape)
        # print('logvar shape:', logvar.shape)
        #       pdb.set_trace()
        return self.decode(z, label), mu, logvar, x_after_mean, z

    def negative_elbo_bound(self, recon, x, mu, var, z):

        prior = self.gaussian_parameters(self.z_init, dim=1)

        # q_m, q_v = self.vae_encode(x)
        # #print("q_m", q_m.size())
        # z_given_x = self.reparameterize(q_m, q_v)
        # decoded_bernoulli_logits = self.decode(z_given_x)
        rec = -self.log_bernoulli_with_logits(recon, x)
        # rec = -torch.mean(rec)
        pdb.set_trace()
        # terms for KL divergence
        log_q_phi = self.log_normal(z, mu, var)
        # print("log_q_phi", log_q_phi.size())
        log_p_theta = self.log_normal_mixture(z, prior[0], prior[1])
        print("log_p_theta", log_p_theta.size())
        kl = log_q_phi - log_p_theta
        # print("kl", kl.size())

        nelbo = torch.mean(kl + rec)

        rec = torch.mean(rec)
        kl = torch.mean(kl)
        return nelbo, rec, kl

    def gaussian_parameters(self, h, dim=-1):
        m, h = torch.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        pdb.set_trace()
        return m, v

    def log_bernoulli_with_logits(self, x, logits):
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        log_prob = -bce(input=logits, target=x).sum(-1)
        return log_prob

    def log_normal(self, x, m, v):
        const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
        log_det = -0.5 * torch.sum(torch.log(v), dim=-1)

        log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)
        log_prob = const + log_det + log_exp
        return log_prob

    def log_normal_mixture(self, z, m, v):
        z = z.unsqueeze(1)
        pdb.set_trace()
        log_probs = self.log_normal(z, m, v)
        log_prob = self.log_mean_exp(log_probs, 1)
        return log_prob

    def log_mean_exp(self, x, dim):
        return self.log_sum_exp(x, dim) - np.log(x.size(dim))

    def log_sum_exp(self, x, dim=0):
        max_x = torch.max(x, dim)[0]
        new_x = x - max_x.unsqueeze(dim).expand_as(x)
        return max_x + (new_x.exp().sum(dim)).log()

    def generate_sample(self, x):
        mu, logvar, _ = self.vae_encode(x)
        #        pdb.set_trace()
        #        pdb.set_trace()
        return Normal(mu, logvar.exp()).sample((1, x[2].shape[2])).squeeze(dim=0)
