""" Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Which was adopated by: Ellis Brown, Max deGroot
    https://github.com/amdegroot/ssd.pytorch

    Further:
    Updated by Gurkirt Singh for ucf101-24 dataset
    Licensed under The MIT License [see LICENSE for details]
"""

import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, UCF24Detection, AnnotationTransform, detection_collate, CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time, copy
from utils.evaluation import evaluate_detections
from layers.box_utils import decode, nms
from utils import  AverageMeter
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, LogLR
from torch.nn.utils import clip_grad_norm
import shutil
from torch.nn import DataParallel
best_prec1 = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

day = (time.strftime('%m-%d',time.localtime(time.time())))
print (day)

def main():
    global my_dict, keys, k_len, arr, xxx, args, log_file, best_prec1

    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
    parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
    parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD')  # only support 300 now
    parser.add_argument('--modality', default='rgb', type=str,
                        help='INput tyep default rgb options are [rgb,brox,fastOF]')
    parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--max_iter', default=120000, type=int, help='Number of training iterations')
    parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--stepvalues', default='70000,90000', type=str,
                        help='iter number when learning rate to be dropped')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.2, type=float, help='Gamma update for SGD')
    parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
    parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
    parser.add_argument('--data_root', default='/data4/lilin/my_code/realtime/', help='Location of VOC root directory')
    parser.add_argument('--save_root', default='/data4/lilin/my_code/realtime/realtime-lstm/saveucf24/', help='Location to save checkpoint models')
    parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
    parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
    parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
    parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')
    parser.add_argument('--clip_gradient', default=40, type=float, help='gradients clip')
    parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=35, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--eval_freq', default=2, type=int, metavar='N', help='evaluation frequency (default: 5)')
    parser.add_argument('--snapshot_pref', type=str, default="ucf101_vgg16_ssd300_gru")
    parser.add_argument('--lr_milestones', default=[-2, -5], type=float, help='initial learning rate')
    parser.add_argument('--arch', type=str, default="VGG16")
    parser.add_argument('--Finetune_SSD', default=True, type=str)

    print(__file__)
    file_name = (__file__).split('/')[-1]
    file_name = file_name.split('.')[0]
    print(file_name)
    ## Parse arguments
    args = parser.parse_args()
    ## set random seeds
    np.random.seed(args.man_seed)
    torch.manual_seed(args.man_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.man_seed)

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    args.cfg = v2
    args.train_sets = 'train'
    args.means = (104, 117, 123)
    num_classes = len(CLASSES) + 1
    args.num_classes = num_classes
    args.stepvalues = [int(val) for val in args.stepvalues.split(',')]
    args.loss_reset_step = 30
    args.eval_step = 10000
    args.print_step = 10
    args.data_root += args.dataset + '/'

    ## Define the experiment Name will used to same directory
    args.snapshot_pref = 'ucf101_CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}_'.format(args.dataset,
                args.modality, args.batch_size, args.basenet[:-14], int(args.lr*100000)) + file_name

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    net = build_ssd(300, args.num_classes, use_gru=True)

    if args.Finetune_SSD is True:
        print ("load snapshot")
        pretrained_weights = "/data4/lilin/my_code/realtime/ucf24/rgb-ssd300_ucf24_120000.pth"
        pretrained_dict = torch.load(pretrained_weights)
        model_dict = net.state_dict()  # 1. filter out unnecessary keys
        pretrained_dict_2 = {k: v for k, v in pretrained_dict.items() if k in model_dict } # 2. overwrite entries in the existing state dict
        # pretrained_dict_2['vgg.25.bias'] = pretrained_dict['vgg.24.bias']
        # pretrained_dict_2['vgg.25.weight'] = pretrained_dict['vgg.24.weight']
        # pretrained_dict_2['vgg.27.bias'] = pretrained_dict['vgg.26.bias']
        # pretrained_dict_2['vgg.27.weight'] = pretrained_dict['vgg.26.weight']
        # pretrained_dict_2['vgg.29.bias'] = pretrained_dict['vgg.28.bias']
        # pretrained_dict_2['vgg.29.weight'] = pretrained_dict['vgg.28.weight']
        # pretrained_dict_2['vgg.32.bias'] = pretrained_dict['vgg.31.bias']
        # pretrained_dict_2['vgg.32.weight'] = pretrained_dict['vgg.31.weight']
        # pretrained_dict_2['vgg.34.bias'] = pretrained_dict['vgg.33.bias']
        # pretrained_dict_2['vgg.34.weight'] = pretrained_dict['vgg.33.weight']
        model_dict.update(pretrained_dict_2) # 3. load the new state dict
    elif args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    elif args.modality == 'fastOF':
        print('Download pretrained brox flow trained model weights and place them at:::=> ',args.data_root + 'ucf24/train_data/brox_wieghts.pth')
        pretrained_weights = args.data_root + 'train_data/brox_wieghts.pth'
        print('Loading base network...')
        net.load_state_dict(torch.load(pretrained_weights))
    else:
        vgg_weights = torch.load(args.data_root +'train_data/' + args.basenet)
        print('Loading base network...')
        net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

    print('Initializing weights for extra layers and HEADs...')
    # initialize newly added layers' weights with xavier method
    if args.Finetune_SSD is False:
        net.module.extras.apply(weights_init)
        net.module.loc.apply(weights_init)
        net.module.conf.apply(weights_init)

    parameter_dict = dict(net.named_parameters()) # Get parmeter of network in dictionary format wtih name being key
    params = []

    #Set different learning rate to bias layers and set their weight_decay to 0
    for name, param in parameter_dict.items():
        if name.find('vgg') > -1 and name.find('cell') <= -1 and int(name.split('.')[1]) < 23:
            param.requires_grad = False
            print(name, 'layer parameters will be fixed')
        else:
            if name.find('bias') > -1:
                print(name, 'layer parameters will be trained @ {}'.format(args.lr*2))
                params += [{'params': [param], 'lr': args.lr*2, 'weight_decay': 0}]
            else:
                print(name, 'layer parameters will be trained @ {}'.format(args.lr))
                params += [{'params':[param], 'lr': args.lr, 'weight_decay':args.weight_decay}]

    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    scheduler = None
    scheduler = LogLR(optimizer, lr_milestones=args.lr_milestones, total_epoch=args.epochs)

    print('Loading Dataset...')
    train_dataset = UCF24Detection(args.data_root, args.train_sets, SSDAugmentation(args.ssd_dim, args.means),
                                   AnnotationTransform(), input_type=args.modality)
    val_dataset = UCF24Detection(args.data_root, 'test', BaseTransform(args.ssd_dim, args.means),
                                 AnnotationTransform(), input_type=args.modality,
                                 full_test=False)

    train_data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    val_data_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, collate_fn=detection_collate, pin_memory=True)

    print ("train epoch_size: ", len(train_data_loader))
    print('Training SSD on', train_dataset.name)

    my_dict = copy.deepcopy(train_data_loader.dataset.train_vid_frame)
    keys = list(my_dict.keys())
    k_len = len(keys)
    arr = np.arange(k_len)
    xxx = copy.deepcopy(train_data_loader.dataset.ids)
    log_file = open(args.save_root + args.snapshot_pref + "_training_" + day + ".log", "w", 1)
    log_file.write(args.snapshot_pref+'\n')

    for arg in vars(args):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')
    log_file.write(str(net))

    torch.cuda.synchronize()
    for epoch in range(args.start_epoch, args.epochs):

        train(train_data_loader, net, criterion, optimizer, scheduler, epoch)

        print('Saving state, epoch:', epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
        }, epoch = epoch)

        #### log lr ###
        scheduler.step()
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            torch.cuda.synchronize()
            tvs = time.perf_counter()
            mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, epoch, iou_thresh=args.iou_thresh)
            # remember best prec@1 and save checkpoint
            is_best = mAP > best_prec1
            best_prec1 = max(mAP, best_prec1)
            print('Saving state, epoch:', epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
            }, is_best,epoch)

            for ap_str in ap_strs:
                print(ap_str)
                log_file.write(ap_str+'\n')
            ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
            print(ptr_str)
            log_file.write(ptr_str)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
            print(prt_str)
            log_file.write(ptr_str)

    log_file.close()


def train(train_data_loader, net, criterion, optimizer, scheduler, epoch):
    net.train()
    # loss counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    #### shuffle ####
    xxxx = copy.deepcopy(train_data_loader.dataset.ids)
    np.random.shuffle(arr)
    iii = 0
    for arr_i in arr:
        key = keys[arr_i]
        rang = my_dict[key]
        xxxx[iii:(iii + rang[1] - rang[0])] = xxx[rang[0]:rang[1]]
        iii += rang[1] - rang[0]
    train_data_loader.dataset.ids = copy.deepcopy(xxxx)

    # create batch iterator
    batch_iterator = None
    iter_count = 0
    t0 = time.perf_counter()

    for iteration in range(len(train_data_loader)):
        if not batch_iterator:
            batch_iterator = iter(train_data_loader)
        # load train data
        images, targets, img_indexs = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        out = net(images, img_indexs)
        # backprop
        optimizer.zero_grad()

        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(net.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        # if scheduler is not None:
        #     scheduler.step()
        loc_loss = loss_l.data[0]
        conf_loss = loss_c.data[0]
        # print('Loss data type ',type(loc_loss))
        loc_losses.update(loc_loss)
        cls_losses.update(conf_loss)
        losses.update((loc_loss + conf_loss) / 2.0)

        if iteration % args.print_step == 0 and iteration > 0:

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_time.update(t1 - t0)

            print_line = 'Epoch {:02d}/{:02d} Iteration {:06d}/{:06d} loc-loss {:.3f}({:.3f}) cls-loss {:.3f}({:.3f}) ' \
                         'average-loss {:.3f}({:.3f}) Timer {:0.3f}({:0.3f}) lr {:0.5f}'.format(
                epoch, args.epochs, iteration, len(train_data_loader), loc_losses.val, loc_losses.avg, cls_losses.val,
                cls_losses.avg, losses.val, losses.avg, batch_time.val, batch_time.avg, args.lr)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            log_file.write(print_line + '\n')
            print(print_line)
            iter_count += 1
            if iter_count % args.loss_reset_step == 0 and iter_count > 0:
                loc_losses.reset()
                cls_losses.reset()
                losses.reset()
                batch_time.reset()
                print('Reset accumulators of ', args.snapshot_pref, ' at', iter_count * args.print_step)
                iter_count = 0


def validate(args, net, val_data_loader, val_dataset, epoch, iou_thresh=0.5):
    """Test a SSD network on an image database."""
    print('Validating at ', epoch)
    num_images = len(val_dataset)
    num_classes = args.num_classes

    det_boxes = [[] for _ in range(len(CLASSES))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    val_step = 100
    count = 0
    net.eval()  # switch net to evaluation mode
    torch.cuda.synchronize()
    ts = time.perf_counter()
    for val_itr in range(len(val_data_loader)-2, len(val_data_loader)):
        if not batch_iterator:
            batch_iterator = iter(val_data_loader)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        images, targets, img_indexs = next(batch_iterator)
        batch_size = images.size(0)
        height, width = images.size(2), images.size(3)

        if args.cuda:
            images = Variable(images.cuda(), volatile=True)

        output = net(images, img_indexs)

        loc_data = output[0]
        conf_preds = output[1]
        prior_data = output[2]

        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            tf = time.perf_counter()
            print('Forward Time {:0.3f}'.format(tf-t1))
        for b in range(batch_size):
            gt = targets[b].numpy()
            gt[:,0] *= width
            gt[:,2] *= width
            gt[:,1] *= height
            gt[:,3] *= height
            gt_boxes.append(gt)
            decoded_boxes = decode(loc_data[b].data, prior_data.data, args.cfg['variance']).clone()
            conf_scores = net.softmax(conf_preds[b]).data.clone()

            for cl_ind in range(1, num_classes):
                scores = conf_scores[:, cl_ind].squeeze()
                c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                scores = scores[c_mask].squeeze()
                # print('scores size',scores.size())
                if scores.dim() == 0:
                    # print(len(''), ' dim ==0 ')
                    det_boxes[cl_ind - 1].append(np.asarray([]))
                    continue
                boxes = decoded_boxes.clone()
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes = boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                scores = scores[ids[:counts]].cpu().numpy()
                boxes = boxes[ids[:counts]].cpu().numpy()
                # print('boxes sahpe',boxes.shape)
                boxes[:,0] *= width
                boxes[:,2] *= width
                boxes[:,1] *= height
                boxes[:,3] *= height

                for ik in range(boxes.shape[0]):
                    boxes[ik, 0] = max(0, boxes[ik, 0])
                    boxes[ik, 2] = min(width, boxes[ik, 2])
                    boxes[ik, 1] = max(0, boxes[ik, 1])
                    boxes[ik, 3] = min(height, boxes[ik, 3])

                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)

                det_boxes[cl_ind-1].append(cls_dets)
            count += 1
        if val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
            torch.cuda.synchronize()
            ts = time.perf_counter()
        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('NMS stuff Time {:0.3f}'.format(te - tf))
    print('Evaluating detections for epoch number ', epoch)
    return evaluate_detections(gt_boxes, det_boxes, CLASSES, iou_thresh=iou_thresh)

def save_checkpoint(state, is_best = False, epoch = 0):
    snapshot = args.save_root + args.snapshot_pref + '_epoch_' + str(epoch)
    filename = snapshot + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        best_name = snapshot + '_model_best.pth.tar'
        shutil.copyfile(filename, best_name)

if __name__ == '__main__':
    main()