import argparse
import os
parser = argparse.ArgumentParser(description='Attention Model')

parser.add_argument('--dataset', default='sysu', help='dataset name: [regdb or sysu]')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--img_h', default=288, type=int, help='img height')
parser.add_argument('--img_w', default=144, type=int, help='img width')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, help='testing batch size')
### directory config
parser.add_argument('--save_path', default='log/', type=str, help='parent save directory')
parser.add_argument('--exp_name', default='exp', type=str, help='child save directory')
### model/training config
parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--save_epoch', default=5, type=int, help='save epochs') # 40
parser.add_argument('--test_every', default=5, type=int, help='test epochs') # 40
parser.add_argument('--lw_dt', default=0.5, type=float, help='weight for dense triplet loss')
parser.add_argument('--margin', default=0.3, type=float, help='triplet loss margin')
parser.add_argument('--method', default='full', type=str, help='method type: [baseline or full]')
### evaluation protocols
parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
### misc
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--nvidia_device', default=0, type=int, help='gpu device to use')
parser.add_argument('--enable_tb', default=True, action='store_true', help='enable tensorboard logging')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.nvidia_device)
