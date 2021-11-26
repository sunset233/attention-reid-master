import argparse
import time

import torchvision.transforms as transforms
from old.dataset import SYSUData
from utils.utils import *

parser = argparse.ArgumentParser(description='Main Model Testing')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')


args = parser.parse_args()
dataset = args.dataset
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
class Session:
    def __init__(self):
        self.log_dir = args.log_dir
        self.model_dir = args.model_dir
        ensure_dir(args.log_dir)
        ensure_dir(args.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
data_path = 'C:/Users/Administrator/Desktop/datasets/SYSU-MM01/'
trainset = SYSUData(data_path, transform = transform_train)
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
n_class = len(np.unique(trainset.train_color_label))
end = time.time()
print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
# print('  ------------------------------')
# print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
# print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))