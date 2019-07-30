import torch
from torch import nn
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

        loss = mel_loss + gate_loss

#        pdb.set_trace()
        for emotion, mean, var in [(x, y, z) for x,y,z in zip(label, mu, logvar)]:
            if torch.equal(emotion, torch.tensor([1,0]).cuda()):
#                print('happy')
                self.mu_happy = mean
                self.var_happy = var
                loss += -0.5 * torch.sum(1 + self.var_happy - self.mu_happy.pow(2) - self.var_happy.exp())
            else:
#                print('sad')
                self.mu_sad = mu
                self.var_sad = logvar
                loss += -0.5 * torch.sum(1 + self.var_sad - self.mu_sad.pow(2) - self.var_sad.exp())

        return loss
