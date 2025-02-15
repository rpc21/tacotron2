import pdb
import pickle
import torch
import argparse
import hparams
from revised_latent_model import GMVAE_revised
from classification import prepare_dataloaders, load_checkpoint
from hparams import create_hparams
from torch.utils.data import DataLoader
import numpy as np


def load_latent_model(hparams, path_to_checkpoint):
    checkpoint = torch.load(path_to_checkpoint)
    #    pdb.set_trace()
    model = GMVAE_revised(hparams, True)
    model.load_state_dict(checkpoint['state_dict'])
    #    model.supervised = True
    return model


def evaluate_latent_model(checkpoint_path):
    model = load_latent_model(hparams, checkpoint_path)
    means = torch.Tensor([]).cuda()
    variances = torch.Tensor([]).cuda()
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # model = load_model(hparams)
    model = GMVAE_revised(hparams).cuda()
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

        with torch.no_grad():
            #        pdb.set_trace()
            model.supervised = True
            x, y = model.parse_batch(batch)
            recon, mu, logvar, x_after_mean = model(x)
            #            pdb.set_trace()
            means = torch.cat((means, mu), dim=0)
            variances = torch.cat((variances, logvar), dim=0)
            #            outputs = torch.cat((outputs, torch.squeeze(recon)), dim=0)
            #        pdb.set_trace()
#            pdb.set_trace()
            print('Iteration {} is complete'.format(iteration))
            iteration += 1

    means = means.mean(dim=0)
    variances = variances.mean(dim=0)
    print(means.shape,variances.shape)
    print(means, variances)
    d = {
        'mean': means,
        'logvar': variances
    }
    with open('/scratch/speech/output/IEMOCAP/fru/means_and_variances.pkl', 'wb+') as f:
        pickle.dump(d, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--latent', type=int, default=1, required=False, help='0 if want to train tacotron')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    evaluate_latent_model('/scratch/speech/output/IEMOCAP/checkpoint_10000')
