from torch import nn
import torch
from torch.nn.functional import binary_cross_entropy, sigmoid, binary_cross_entropy_with_logits
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
#    print('shape of recon_x:', recon_x.transpose(1,2).shape)
 #   print('shape of x:', x[-1].shape)
  #  print('shape of mu:', mu.shape)
   # print('shape of logvar:', logvar.shape)
#    val=torch.max(x[-1])
 #   x=x[-1]/val
  #  val1=(recon_x.transpose(1,2)).max()
   # recon_x=recon_x.transpose(1,2)/val1
    #x_mean=x.mean()
    #x_std=x.std()
   # x_n=(x- x_mean)/x_std
  #  recon_x_mean=recon_x.mean()
 #   recon_x_std=recon_x.std()
#    recon_x_n=(recon_x-recon_x_mean)/recon_x_std
    #pdb.set_trace()
   # with torch.no_grad():
    #    y = x_n
    loss = binary_cross_entropy(recon_x, torch.sigmoid(x).detach()) #, reduction='sum')
#    print("BCE:", BCE)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#    m = nn.LogSoftmax(dim=1)
#    mse = nn.MSELoss()
 #   loss = mse(recon_x,x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#    print("KLD:", KLD)
    return loss + KLD
