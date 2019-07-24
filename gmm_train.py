import os
import sys
import time
import argparse
import math
import wavio
import pdb
import numpy as np
from numpy import finfo
from waveglow.glow import WaveGlow

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import GMVAE
from IEMOCAP_data_utils import IEMOCAPLoader, TextMelCollate
from loss_function import Tacotron2Loss, GMVAELoss
from logger import Tacotron2Logger
from hparams import create_hparams
from class_t import GMVAE_revised
from text import text_to_sequence


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = IEMOCAPLoader(hparams.training_files, hparams)
    valset = IEMOCAPLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


'''def load_model(hparams):
    model = Tacotron2(hparams, True).cuda()
    if hparams.fp16_run:
        model.decoder_enhanced.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model'''


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict, strict=False)
    print('loaded pretrained model')
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


'''def make_inferences(model, iteration, hparams, output_directory):
    output_directory = output_directory + 'checkpoint_{}_samples/'.format(iteration)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    waveglow_path = '/scratch/speech/waveglow_256channels.pt'
    sys.path.append('waveglow/')
#    pdb.set_trace()
    waveglow = torch.load(waveglow_path)['model']
    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    model.eval()
    sentences = [
        "Waveglow is really awesome!",
        "Data plus is super fun",
        "This is a test sentence",
        "This code keeps breaking",
        "Decrease already loss function",
        "This is a much longer text file to see if the length of the sentence causes it to break",
        "Word"
    ]
    for i, text in enumerate(sentences):
        #### Just for testing purposes
        text = "Waveglow is really awesome!"
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

        wavio.write(output_directory + 'sentence_{}.wav'.format(i), audio[0].data.cpu().numpy(), rate=float(hparams.sampling_rate), sampwidth=3)
        ##########

    print('generated samples')
    model.train()'''


'''def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, hparams=None, output_directory=None):
    """Handles all the validation scoring and printing"""
#    make_inferences(model, iteration, hparams, output_directory)

    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    print('Making inferences')
    make_inferences(model, iteration, hparams, output_directory)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(reduced_val_loss, model, y, y_pred, iteration)'''


def validate_latent(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, output_directory, hparams):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            recon, mu, logvar, x_after_mean, z = model(x)

            loss, recon, kl = criterion(recon, x_after_mean, mu, logvar, z)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        with open(output_directory + 'output_stats.txt', 'a+') as f:
            f.write("Validation loss {}: {:9f}  \n".format(iteration, reduced_val_loss))

def train_latent(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

       Params
       ------
       output_directory (string): directory to save checkpoints
       log_directory (string) directory to save tensorboard logs
       checkpoint_path(string): checkpoint path
       n_gpus (int): number of gpus
       rank (int): rank of current gpu
       hparams (object): comma separated list of "name=value" pairs.
       """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # model = load_model(hparams)
    model = GMVAE_revised(hparams, True).cuda()
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    # criterion = Tacotron2Loss()
    criterion = model.negative_elbo_bound

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    print('===========================Number of parameters===================================')
    print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            recon, mu, logvar, x_after_mean, z = model(x)

            loss, recon, kl = criterion(recon, x_after_mean, mu, logvar, z)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            print("Epoch {}: Batch Size: {} Train loss {} {:.6f} Grad Norm {:.6f} \n".format(
                epoch, hparams.batch_size, iteration, reduced_loss, grad_norm))

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                with open(output_directory + 'output_stats.txt', 'a+') as f:
                    f.write("Epoch {}: Batch Size: {} Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it\n".format(
                        epoch, hparams.batch_size, iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate_latent(model, criterion, valset, iteration,
                                hparams.batch_size, n_gpus, collate_fn, logger,
                                hparams.distributed_run, rank, output_directory, hparams)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1
'''def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))


    model.train()
    is_overflow = False
    skipped = 0
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)

            try:
                y_pred = model(x)
            except ValueError:
                skipped += 1
                print('Skipped an iteration due to value error, you have now skipped {} iterations'.format(skipped))
                iteration += 1
                continue

            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0) and iteration > 0:
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, hparams=hparams, output_directory=output_directory)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1'''
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

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

#    if args.latent==1:
#    hparams.batch_size = 1
    train_latent(args.output_directory, args.log_directory, args.checkpoint_path,
           args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
#    else:
#    hparams.batch_size = 8
#     train(args.output_directory, args.log_directory, args.checkpoint_path,
#          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
