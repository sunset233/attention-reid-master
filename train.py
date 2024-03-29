import sys
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.data_utils import *
from utils.eval_utils import eval_regdb, eval_sysu
from utils.loss import OriTripletLoss, ContrastiveLoss
from utils.misc import AverageMeter,Logger, set_seed
from model import network
from settings import parser
import torch.optim as optim
from torch.autograd import Variable
import time
import os
import os.path as osp

args = parser.parse_args()

if args.enable_tb:
    from tensorboardX import SummaryWriter

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/home/lxz/lph/datasets/SYSU-MM01/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/home/lxz/lph/datasets/RegDB/'
    test_mode = [2, 1]  # visible to thermal

log_path = osp.join(args.save_path, args.exp_name)
checkpoint_path = osp.join(log_path, 'checkpoints')

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

sys.stdout = Logger(os.path.join(log_path, 'os.txt'))

# tensorboard
if args.enable_tb:
    vis_log_dir = osp.join(log_path, 'vis_log')
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Loading data..')
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')

net = network(n_class, args)
net.to(device)
cudnn.benchmark = True

crtierion_id = nn.CrossEntropyLoss()
critierion_tri = OriTripletLoss(batch_size=args.batch_size * args.num_pos, margin=args.margin)
critierion_con = ContrastiveLoss(batch_size=args.batch_size * args.num_pos, temperature=args.temperature)

crtierion_id.to(device)
critierion_tri.to(device)
critierion_con.to(device)

ignored_params = list(map(id, net.bottleneck.parameters())) \
    + list(map(id, net.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer = optim.SGD([
    {'params': base_params, 'lr': 0.1 * args.lr},
    {'params': net.bottleneck.parameters(), 'lr': args.lr},
    {'params': net.classifier.parameters(), 'lr': args.lr}],
    weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr
def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss_meter = AverageMeter()
    loss_id_meter = AverageMeter()
    # if args.method == 'full':
    #     loss_con_meter = AverageMeter()


    data_time = AverageMeter()
    batch_time = AverageMeter()

    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, data in enumerate(trainloader):

        img_v, img_t = data['img_v'], data['img_t']
        target_v, target_t = data['target_v'], data['target_t']

        labels = torch.cat((target_v, target_t), 0)

        img_v = Variable(img_v.cuda())
        img_t = Variable(img_t.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)
        out = net(img_v, img_t)

        x_p = out['x_p']
        loss_tri, batch_acc = critierion_tri(out['feat_p'], labels)
        loss_id = crtierion_id(out['cls_id'], labels) + loss_tri
        # loss_con = critierion_con(x_p, feat_p)
        loss = loss_tri
        # if args.method == 'full':
        #     loss = loss_id + loss_tri
        # else:
        #     loss = loss_id
        correct += (batch_acc / 2)
        _, predicted = out['cls_id'].max(1)
        correct += (predicted.eq(labels).sum().item() / 2)
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_meter.update(loss.item(), 2 * img_v.size(0))
        loss_id_meter.update(loss_id.item(), 2 * img_v.size(0))
        # if args.method == 'full':
        #     loss_con_meter.update(loss_con.item(), 2 * img_v.size(0))

        total += labels.size(0)
        # reid master
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            if args.method == 'full':
                print('Epoch: [{}][{}/{}] '
                      'lr: {:.3f} '
                      'loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'loss_id: {loss_id.val:.4f} ({loss_id.avg:.4f}) '
                      # 'loss_con: {loss_con.val:.4f} ({loss_con.avg:.4f})'
                    .format(
                    epoch, batch_idx, len(trainloader),
                    current_lr,
                    train_loss=train_loss_meter,
                    loss_id=loss_id_meter)
                    # loss_con=loss_con_meter
                )
            else:
                print('Epoch: [{}][{}/{}] '
                      'lr: {:.3f} '
                      'loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'loss_id: {loss_id.val:.4f} ({loss_id.avg:.4f}) '
                      'acc: {acc:.2f}'.format(
                    epoch, batch_idx, len(trainloader),
                    current_lr,
                    train_loss=train_loss_meter,
                    loss_id=loss_id_meter,
                    acc=100. * correct / total)
                )

    if args.enable_tb:
        writer.add_scalar('total_loss', train_loss_meter.avg, epoch)
        writer.add_scalar('id_loss', loss_id_meter.avg, epoch)
        writer.add_scalar('lr', current_lr, epoch)

        writer.add_scalar('train_acc', 100. * correct / total, epoch)
        # if args.method == 'full':
        #     writer.add_scalar('contrast_loss', loss_con_meter.avg, epoch)

def evaluation(epoch):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, data in enumerate(gall_loader):
            input = data['img']
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input, input, test_mode[0])['feat_p']
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, data in enumerate(query_loader):
            input = data['img']
            batch_num = input.size(0) # round 59
            input = Variable(input.cuda())
            print('batch_index: size:', batch_idx, input.size(0))
            feat = net(input, input, test_mode[0])['feat_p']
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    if args.enable_tb:
        writer.add_scalar('rank1', cmc[0], epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('mINP', mINP, epoch)

    return cmc, mAP, mINP

# training
print('==> Start Training...')
for epoch in range(start_epoch, 81 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size*args.num_pos, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % args.test_every == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP = evaluation(epoch)
        # save model
        if cmc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, osp.join(checkpoint_path,'best.t'))

        # save model
        if epoch > 5 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, osp.join(checkpoint_path, 'ep_{:02d}.t'.format(epoch)))

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('Best Epoch [{}]'.format(best_epoch))
