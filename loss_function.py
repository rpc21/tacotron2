import torch
from torch import nn


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


class GMMVAELoss(nn.Module):
    def __init__(self):
        super(GMMVAELoss, self).__init__()
        self.mu_happy = torch.Tensor()
        self.var_happy = torch.Tensor()
        self.mu_sad = torch.Tensor()
        self.var_sad = torch.Tensor()

    def forward(self, model_output, targets, mu, logvar, label):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        if label == torch.tensor([1,0]):
            self.mu_happy = mu
            self.var_happy = logvar
            KLD = -0.5 * torch.sum(1 + self.var_happy - self.mu_happy.pow(2) - self.var_happy.exp())
        else:
            self.mu_sad = mu
            self.var_sad = logvar
            KLD = -0.5 * torch.sum(1 + self.var_sad - self.mu_sad.pow(2) - self.var_sad.exp())

        return mel_loss + gate_loss + KLD, KLD