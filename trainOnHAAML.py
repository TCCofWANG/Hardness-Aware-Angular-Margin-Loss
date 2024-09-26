# -*- coding: utf-8 -*-
'''
Quantify sample difficulty using the difference between negative class cosine and margined positive class cosine.
'''
import os
import torch.utils.data
import torch.nn as nn
from torch.nn import DataParallel
from datetime import datetime
from utils.logging import init_log

from backbone.ResNet import *
from backbone.IR_SE_ResNet import *
from backbone.mobilenet import *
from backbone.mobilenetV2 import *

from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from tensorboardX import SummaryWriter
from data.val_dataset import ValDataset
from utils.utils import make_weights_for_balanced_classes, get_val_data, perform_val, buffer_val, setup_seed, perform_val_

import math
import torch
from torch.nn import Parameter
import torch.nn.functional as F

class HAAML(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=64.0, m_0=0.50, t=1.2, loss_weight = 10):
        super(HAAML, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m_0 = m_0
        self.t = t
        self.loss_weight = loss_weight
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))#num_class*feat_dim
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m_0)
        self.sin_m = math.sin(m_0)


    def forward(self, x, label):
        # cos(theta)
        x_norm = x.renorm(2,0,1e-5).mul(1e5)
        n_weight = self.weight.renorm(2,0,1e-5).mul(1e5)
        cos_theta = F.linear(x_norm, n_weight)
        cos_theta = cos_theta.clamp(-1, 1)

        # ground true
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)

        # take cos(theta + m) into cos_theta
        sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
        cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m
        # cos_theta_m = torch.cos(torch.acos(gt) + self.m_0)

        # new_m
        hard_mask = (cos_theta > cos_theta_m).type(torch.FloatTensor).cuda()
        hard_mask.scatter_(1, label.view(-1, 1), 0)
        hard_cos = torch.where(hard_mask > 0, cos_theta-cos_theta_m, torch.zeros_like(cos_theta))
        # hard_cos_one = torch.where(hard_mask > 0, torch.ones_like(cos_theta), torch.zeros_like(cos_theta))
        hard_cos_num = torch.sum(hard_mask, dim=1).view(-1,1)
        hard_level = torch.sum(hard_cos, dim=1).view(-1,1)
        hard_cos_num = hard_cos_num.clamp(1, self.out_feature) # avoid /0
        H = hard_level/hard_cos_num
        with torch.no_grad():
            new_m = self.m_0 + self.t * torch.log(H + 1)
            new_m = torch.where(new_m > 0.75, torch.zeros_like(new_m), new_m)
            cos_new_m = torch.cos(new_m)
            sin_new_m = torch.sin(new_m)

        # cos(theta + new_m)
        cos_theta_newm = gt * cos_new_m - sin_theta * sin_new_m
        # new_gt = torch.where(gt > 0, cos_theta_newm , gt) # easy_margin=true 2024.6.5

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        threshold = torch.cos(math.pi - new_m)
        mm = torch.sin(math.pi - new_m) * new_m
        new_gt = torch.where(gt > threshold, cos_theta_newm, gt - mm) # easy_margin=false

        cos_theta.scatter_(1, label.view(-1, 1), new_gt)
        output = cos_theta * self.s

        # regularizer
        hard_regularizer = self.loss_weight * torch.mean(H)

        return output, hard_regularizer, gt, new_m.view(1, -1), H.view(1, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    # parm for model
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--data_train', type=str, default='CASIA_WF',
                        help='data used to train : MS1MV2, CASIA_WF, MS1MV3')
    parser.add_argument('--backbone', type=str, default='ir_18',
                        help='res_18, res_34, res_50, res_101, res_152,'
                             'ir_18, ir_34, ir_50, ir101, ir_se_50, ir_se_100'
                             'mobilenet, mobilenetv2')
    parser.add_argument('--loss_function', type=str, default='HAAML')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float, default=64.0, help='scale size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--total_epochs', type=int, default=32, help='total epochs')
    # parm for train
    parser.add_argument('--seed_list', type=int, default=[1024], help='random seed for reproduce results')
    parser.add_argument('--gpus', type=str, default='5', help='cuda ID')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--resume', type=bool, default=False, help='remember to change the root in line_222')
    parser.add_argument('--ckpt_path', type=str, default='./new_ckpt',
                        help='resume model')
    parser.add_argument('--log_dir', type=str, default='./new_train_log', help='model save dir')
    parser.add_argument('--step', type=str, default='8,14,20', help='step settings')
    # parm for test
    parser.add_argument('--tta', type=bool, default=True, help='Whether use Data Augmentation')
    # parm for train
    parser.add_argument('--m_0', type=float, default=0.5, help='base margin')
    parser.add_argument('--t', type=float, default=0.4, help='base margin')
    parser.add_argument('--loss_weight', type=float, default=10, help='base margin')
    args = parser.parse_args()
    args.milestones = [int(p) for p in args.step.split(',')]

    # log_init
    save_dir = os.path.join(args.log_dir,
                            args.data_train + '_' + args.backbone.upper() + '_' + args.loss_function + '_' + datetime.now().strftime(
                                '%Y%m%d_%H%M%S'))
    save_ckpt_dir = os.path.join(args.ckpt_path,
                                 args.data_train + '_' + args.backbone.upper() + '_' + args.loss_function + '_' + datetime.now().strftime(
                                     '%Y%m%d_%H%M%S'))

    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    # tensorboard
    writer = SummaryWriter(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    for seed in args.seed_list:

        # set the seed for repeating exp
        setup_seed(seed)

        # gpu_init
        multi_gpus = False
        if len(args.gpus.split(',')) > 1:
            multi_gpus = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("gpu:", args.gpus)
        # dataset loader
        transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.RandomCrop([112, 112]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

        # train data
        trainset = datasets.ImageFolder(os.path.join(args.data_root, args.data_train), transform)
        # create a weighted random sampler to process imbalanced data
        weights = make_weights_for_balanced_classes(trainset.imgs, len(trainset.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,pin_memory=True, sampler=sampler, num_workers=20, drop_last=True)
        NUM_CLASS = len(trainloader.dataset.classes)
        print("Number of Training Classes: {}    Number of iters per epoch: {}".format(len(trainset.classes),
                                                                                       len(trainloader)))

        # test data
        lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame = get_val_data(
            args.data_root)

        print("=" * 60)
        print("Preparing the val data...")
        lfw_dataset = ValDataset(lfw, 'lfw')
        # cfp_ff_dataset = ValDataset(cfp_ff,'cfp_ff')
        cfp_fp_dataset = ValDataset(cfp_fp,'cfp_fp')
        agedb_dataset = ValDataset(agedb,'agedb')
        calfw_dataset = ValDataset(calfw,'calfw')
        cplfw_dataset = ValDataset(cplfw,'cplfw')

        lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                                 drop_last=False, pin_memory=True)
        cfp_fp_loader = torch.utils.data.DataLoader(cfp_fp_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                                 drop_last=False, pin_memory=True)
        agedb_loader = torch.utils.data.DataLoader(agedb_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                                 drop_last=False, pin_memory=True)
        calfw_loader = torch.utils.data.DataLoader(calfw_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                                 drop_last=False, pin_memory=True)
        cplfw_loader = torch.utils.data.DataLoader(cplfw_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                                 drop_last=False, pin_memory=True)

        # backbone
        print("=" * 60)
        print('building the model...')
        if args.backbone == 'res_18':
            net = ResNet18(feature_dim=args.feature_dim)
        elif args.backbone == 'res_34':
            net = ResNet34(feature_dim=args.feature_dim)
        elif args.backbone == 'res_50':
            net = ResNet50(feature_dim=args.feature_dim)
        elif args.backbone == 'ir_18':
            net = IR_18(feature_dim=args.feature_dim)
        elif args.backbone == 'ir_34':
            net = IR_34(feature_dim=args.feature_dim)
        elif args.backbone == 'ir_50':
            net = IR_50(feature_dim=args.feature_dim)
        elif args.backbone == 'ir_se_50':
            net = IR_SE_50(feature_dim=args.feature_dim)
        elif args.backbone == 'ir_se_100':
            net = IR_SE_101(feature_dim=args.feature_dim)
        elif args.backbone == 'mobilenet':
            net = MobileNet(feat_dim=args.feature_dim)
        elif args.backbone == 'mobilenetv2':
            net = MobileNetV2(feat_dim=args.feature_dim)
        else:
            print(args.backbone, ' is not available!')
        _print("{} Backbone Generated".format(args.backbone))
        _print("=" * 60)

        # loss_function & Hard-Mining function
        loss_function = HAAML(args.feature_dim, len(trainset.classes), s=args.scale_size, m_0=args.m_0, t = args.t, loss_weight=args.loss_weight)
        _print("{} loss Generated".format(args.loss_function))
        _print("Loss parm:m_0 = {},t = {}, loss_weight = {}".format(loss_function.m_0, loss_function.t, loss_function.loss_weight))
        _print("=" * 60)

        # define optimizers for different layer
        optimizer = optim.SGD([
            {'params': net.parameters(), 'weight_decay': 5e-4},
            {'params': loss_function.parameters(), 'weight_decay': 5e-4}
        ], lr=args.lr, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # set gpus
        if multi_gpus:
            net = DataParallel(net).to(device)
            # loss_function = DataParallel(loss_function).to(device)

        else:
            net = net.to(device)
        loss_function = loss_function.to(device)

        # initialize metric
        best_lfw_acc = 0.0
        best_lfw_epoch = 0
        best_agedb_acc = 0.0
        best_agedb_epoch = 0
        best_cfp_fp_acc = 0.0
        best_cfp_fp_epoch = 0
        best_calfw_acc = 0.0
        best_calfw_epoch = 0
        best_cplfw_acc = 0.0
        best_cplfw_epoch = 0
        best_avg_acc = 0.0
        best_avg_epoch = 0
        total_iters = 0
        start_epoch = 0


        # ckpt resume
        if args.resume:
            checkpoint = torch.load(
                '/public/cjp/Pycharmproject/FR_new/ckpt/CASIA_WF_RES_18_ArcFace_20230315_112144/BatchSize256_Seed1023_checkpoint.pth')  # �}��
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            exp_lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
            if multi_gpus:
                net.module.load_state_dict(torch.load(
                    '/public/cjp/Pycharmproject/FR_new/ckpt/CASIA_WF_RES_18_ArcFace_20230315_112144/BatchSize256_Seed1023_net.pth'))
                loss_function.module.load_state_dict(torch.load(
                    '/public/cjp/Pycharmproject/FR_new/ckpt/CASIA_WF_RES_18_ArcFace_20230315_112144/BatchSize256_Seed1023_loss.pth'))
            else:
                net.load_state_dict(torch.load(
                    '/public/cjp/Pycharmproject/FR_new/ckpt/CASIA_WF_RES_18_ArcFace_20230315_112144/BatchSize256_Seed1023_net.pth'))
                loss_function.load_state_dict(torch.load(
                    '/public/cjp/Pycharmproject/FR_new/ckpt/CASIA_WF_RES_18_ArcFace_20230315_112144/BatchSize256_Seed1023_loss.pth'))

        for epoch in range(start_epoch + 1, args.total_epochs + 1):
            list_hardness = []
            list_margin = []
            # Train model
            _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epochs))
            net.train()
            total, correct = 0, 0
            since = time.time()
            if(epoch<=8): #warm
                loss_function.t = 0.0
                loss_function.loss_weight = 0.0

            else:
                loss_function.t = args.t
                loss_function.loss_weight = args.loss_weight

            _print("Loss parm:m_0 = {},t = {}, loss_weight = {}".format(loss_function.m_0, loss_function.t,
                                                                       loss_function.loss_weight))

            for t,data in enumerate(trainloader):
                img, label = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                raw_logits = net(img)
                output, hard_regular, cos_theta, new_m, hardness = loss_function(raw_logits, label)
                loss = criterion(output, label) + hard_regular.mean()
                loss.backward()
                optimizer.step()
                list_margin.extend(new_m.detach().cpu().numpy().flatten())
                list_hardness.extend(hardness.detach().cpu().numpy().flatten())

                # save train acc per class
                _, predict = output.max(1)
                correct += predict.eq(label).sum().item()
                total += len(label)
                total_iters += 1
                writer.add_scalar("Training_Loss", loss.item(), total_iters)

                # print train information
                if total_iters % 500 == 0:
                    acc = correct / total
                    time_cur = (time.time() - since) / 500
                    since = time.time()
                    # print(m_1)
                    _print(
                        "Iters: {:0>6d}/[{:0>2d}], total_loss: {:.4f}, hard_loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(
                            total_iters, epoch, loss.item(), hard_regular.mean(),
                            acc * 100, time_cur, exp_lr_scheduler.get_last_lr()[0]))
            exp_lr_scheduler.step()

            list_margin = np.array(list_margin)
            list_hardness = np.array(list_hardness)
            np.save(save_dir + '/margin_{}.npy'.format(epoch), list_margin)
            np.save(save_dir + '/hardness_{}.npy'.format(epoch), list_hardness)
            # use Tensorboard to plot the curve

            writer.add_scalar("Training_Accuracy", acc * 100, epoch)

            # Test  #int(args.total_epochs / 3):
            if epoch > int(args.total_epochs / 2):
                print("Perform Evaluation on LFW, CFP_FP, AgeDB, CALFW, CPLFW , and Save Checkpoints...")
                # LFW
                accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val_(multi_gpus, device, args.feature_dim,
                                                                              net, lfw_loader, lfw_issame, tta=args.tta)
                buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch)
                _print('LFW Ave Accuracy: {:.4f}'.format(accuracy_lfw * 100))
                if best_lfw_acc <= accuracy_lfw * 100:
                    best_lfw_acc = accuracy_lfw * 100
                    best_lfw_epoch = epoch
                # CFP_FP
                accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val_(multi_gpus, device,args.feature_dim,
                                                                                        net, cfp_fp_loader, cfp_fp_issame, tta=args.tta)
                buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch)
                _print('CFP_FP Ave Accuracy: {:.4f}'.format(accuracy_cfp_fp * 100))
                if best_cfp_fp_acc <= accuracy_cfp_fp * 100:
                    best_cfp_fp_acc = accuracy_cfp_fp * 100
                    best_cfp_fp_epoch = epoch
                # AGE_DB
                accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val_(multi_gpus, device,args.feature_dim, net, agedb_loader,
                                                                                    agedb_issame, tta=args.tta)
                buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch)
                _print('AgeDB Ave Accuracy: {:.4f}'.format(accuracy_agedb * 100))
                if best_agedb_acc <= accuracy_agedb * 100:
                    best_agedb_acc = accuracy_agedb * 100
                    best_agedb_epoch = epoch
                # CALFW
                accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val_(multi_gpus, device,args.feature_dim, net, calfw_loader,
                                                                                    calfw_issame, tta=args.tta)
                buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch)
                _print('CALFW Ave Accuracy: {:.4f}'.format(accuracy_calfw * 100))
                if best_calfw_acc <= accuracy_calfw * 100:
                    best_calfw_acc = accuracy_calfw * 100
                    best_calfw_epoch = epoch
                # CPLFW
                accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val_(multi_gpus, device,args.feature_dim, net, cplfw_loader,
                                                                                    cplfw_issame, tta=args.tta)
                buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch)
                _print('CPLFW Ave Accuracy: {:.4f}'.format(accuracy_cplfw * 100))
                if best_cplfw_acc <= accuracy_cplfw * 100:
                    best_cplfw_acc = accuracy_cplfw * 100
                    best_cplfw_epoch = epoch
                # the best avg_acc
                avg_acc = (accuracy_lfw + accuracy_cfp_fp + accuracy_calfw + accuracy_cplfw + accuracy_agedb) / 5
                if best_avg_acc <= avg_acc * 100:
                    best_avg_acc = avg_acc * 100
                    best_avg_epoch = epoch
                    # save the best epoch
                    msg = 'Saving the best epoch checkpoint: {}'.format(epoch)
                    _print(msg)
                    if multi_gpus:
                        checkpoint = {
                            'optimizer': optimizer.state_dict(),
                            "epoch": epoch,
                            'lr_schedule': exp_lr_scheduler.state_dict()
                        }
                        net_state_dict = net.module.state_dict()
                        # loss_state_dict = loss_function.module.state_dict()
                        loss_state_dict = loss_function.state_dict()
                        if not os.path.exists(save_ckpt_dir):
                            os.mkdir(save_ckpt_dir)
                        torch.save(net_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                "BatchSize{}_Seed{}_net_best.pth".format(args.batch_size, seed)))
                        torch.save(loss_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_loss_best.pth'.format(args.batch_size, seed)))
                        torch.save(checkpoint, os.path.join(save_ckpt_dir,
                                                            'BatchSize{}_Seed{}_checkpoint_best.pth'.format(args.batch_size,
                                                                                                       seed)))
                    else:
                        checkpoint = {
                            'optimizer': optimizer.state_dict(),
                            "epoch": epoch,
                            'lr_schedule': exp_lr_scheduler.state_dict()
                        }
                        net_state_dict = net.state_dict()
                        loss_state_dict = loss_function.state_dict()
                        if not os.path.exists(save_ckpt_dir):
                            os.mkdir(save_ckpt_dir)
                        torch.save(net_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_net_best.pth'.format(args.batch_size, seed)))
                        torch.save(loss_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_loss_best.pth'.format(args.batch_size, seed)))
                        torch.save(checkpoint, os.path.join(save_ckpt_dir,
                                                            'BatchSize{}_Seed{}_checkpoint_best.pth'.format(args.batch_size,
                                                                                                       seed)))

                _print(
                    "Best average accuracy of 5 test datasets: Avg_acc: {:.4f} in epoch: {}".format(best_avg_acc,
                                                                                                    best_avg_epoch))
                # the best acc of each dataset
                _print(
                    'Current Best Accuracy: LFW: {:.4f} in epoch: {}, CFP-FP: {:.4f} in epoch: {} , AgeDB-30: {:.4f} in epoch: {} , CALFW: {:.4f} in epoch: {} and CPLFW: {:.4f} in epoch: {}'.format(
                        best_lfw_acc, best_lfw_epoch, best_cfp_fp_acc, best_cfp_fp_epoch,
                        best_agedb_acc, best_agedb_epoch, best_calfw_acc, best_calfw_epoch, best_cplfw_acc,
                        best_cplfw_epoch))
                # save ckpt every epoch
                if epoch % args.save_freq == 0:
                    msg = 'Saving checkpoint: {}'.format(epoch)
                    _print(msg)
                    if multi_gpus:
                        checkpoint = {
                            'optimizer': optimizer.state_dict(),
                            "epoch": epoch,
                            'lr_schedule': exp_lr_scheduler.state_dict()
                        }
                        net_state_dict = net.module.state_dict()
                        loss_state_dict = loss_function.state_dict()
                        if not os.path.exists(save_ckpt_dir):
                            os.mkdir(save_ckpt_dir)
                        torch.save(net_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                "BatchSize{}_Seed{}_net_now.pth".format(args.batch_size, seed)))
                        torch.save(loss_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_loss_now.pth'.format(args.batch_size, seed)))
                        torch.save(checkpoint, os.path.join(save_ckpt_dir,
                                                            'BatchSize{}_Seed{}_checkpoint_now.pth'.format(args.batch_size,
                                                                                                       seed)))
                    else:
                        checkpoint = {
                            'optimizer': optimizer.state_dict(),
                            "epoch": epoch,
                            'lr_schedule': exp_lr_scheduler.state_dict()
                        }
                        net_state_dict = net.state_dict()
                        loss_state_dict = loss_function.state_dict()
                        if not os.path.exists(save_ckpt_dir):
                            os.mkdir(save_ckpt_dir)
                        torch.save(net_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_net_now.pth'.format(args.batch_size, seed)))
                        torch.save(loss_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_loss_now.pth'.format(args.batch_size, seed)))
                        torch.save(checkpoint, os.path.join(save_ckpt_dir,
                                                            'BatchSize{}_Seed{}_checkpoint_now.pth'.format(args.batch_size,
                                                                                                       seed)))

        _print('Finish training'+'_'+str(seed))