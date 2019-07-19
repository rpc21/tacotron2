import argparse

import torch

from classification import train
from hparams import create_hparams

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
#    train_latent(args.output_directory, args.log_directory, args.checkpoint_path,
#           args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
#    else:
#    hparams.batch_size = 8
train(args.output_directory, args.log_directory, args.checkpoint_path,
     args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
