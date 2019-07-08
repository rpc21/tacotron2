from torch import nn
import torch
from torch.nn.functional import binary_cross_entropy
import pdb


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss




class GMVAELoss(nn.Module):
    def __init__(self):
        super(GMVAELoss, self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        return elbo_loss_function(recon_x, x, mu, logvar)


def elbo_loss_function(recon_x, x, mu, logvar):
#    pdb.set_trace()
#    BCE = binary_cross_entropy(recon_x.transpose(1,2), x[-1], reduction='sum')
#    print("BCE:", BCE)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE1=nn.MSELoss()
    BCE=BCE1(recon_x.transpose(1,2),x[-1])
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return 128*BCE + KLD


    return loss(recon_x, x)
