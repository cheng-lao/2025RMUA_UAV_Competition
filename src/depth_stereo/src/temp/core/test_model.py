import torch 
import os
import psutil
import argparse
from tc_stereo import TCStereo
pid = os.getpid()
process = psutil.Process(pid)
nice = process.nice(None)


parser = argparse.ArgumentParser()
parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
parser.add_argument('--dataset', help="dataset for evaluation", required=True,
                    choices=["kitti", "things", "TartanAir"])
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
parser.add_argument('--uncertainty_threshold', default=0.5, type=float, help='the threshold of uncertainty')
parser.add_argument('--visualize', action='store_true', help='visualize the results')
parser.add_argument('--device', default=0, type=int, help='the device id')

# Architecure choices
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--temporal', action='store_true', help="temporal mode")  # TODO: MODEL temporal mode

args = parser.parse_args()

# if args.visualize:
# wandb.init(
#     job_type="test",
#     project="vis",
#     entity="zengjiaxi"
# )
# add the args to wandb
# wandb.config.update(args)
model = TCStereo(args)
if args.restore_ckpt is not None:
    assert args.restore_ckpt.endswith(".pth")
    checkpoint = torch.load(args.restore_ckpt)
    model.load_state_dict(checkpoint['model'], strict=True)

model = model.cuda(args.device)