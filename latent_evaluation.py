import pdb

import torch

import hparams
from model import GMVAE
from train import prepare_dataloaders, load_checkpoint


def load_latent_model(hparams, path_to_checkpoint):
    checkpoint = torch.load(hparams, path_to_checkpoint)
    model = GMVAE(hparams)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def evaluate_latent_model(checkpoint_path):
    model = load_latent_model(hparams, checkpoint_path)
    inputs = []
    outputs = []
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # model = load_model(hparams)
    model = GMVAE(hparams).cuda()
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    iteration += 1  # next iteration is iteration + 1
    epoch_offset = max(0, int(iteration / len(train_loader)))

    model.eval()
    print('===========================Number of parameters===================================')
    print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    is_overflow = False
    for i, batch in enumerate(train_loader):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        model.zero_grad()
        x, y = model.parse_batch(batch)
        recon, mu, logvar, x_after_mean = model(x)
        pdb.set_trace()

        print('Iteration {} is complete')
        iteration += 1


if __name__ == '__main__':
    evaluate_latent_model('/scratch/speech/output/IEMOCAP/checkpoint_330000')