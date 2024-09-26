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
from utils.utils import make_weights_for_balanced_classes, get_val_data, perform_val, buffer_val, setup_seed, perform_val_
# from data.val_dataset import ValDataset
from data.val_dataset import ValDataset
from data.train_dataset import CASIAWebFace
import math
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from scipy.io import savemat

class ArcFace(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=64.0, m=0.50, easy_margin=True):
        super(ArcFace, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        nweight = self.weight.renorm(2, 0, 1e-5).mul(1e5)
        feature = x.renorm(2, 0, 1e-5).mul(1e5)
        cosine = F.linear(feature, nweight)
        cosine = cosine.clamp(-1, 1)
        gt = cosine[torch.arange(0, label.size(0)), label].view(-1, 1)
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output = output * self.s


        return output, gt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    # parm for model
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--train_file_list', type=str, default='')
    parser.add_argument('--data_train', type=str, default='C_10_O_0_CASIA_WF',
                        help='data used to train : C_?O_?_CASIA_WF')
    parser.add_argument('--backbone', type=str, default='ir_50',
                        help='res_18, res_34, res_50, res_101, res_152,'
                             'ir_18, ir_34, ir_50, ir101, ir_se_50, ir_se_100'
                             'mobilenet, mobilenetv2')
    parser.add_argument('--loss_function', type=str, default='ArcFace_Laplace')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--total_epochs', type=int, default=32, help='total epochs')
    parser.add_argument('--epsilon', default=1e-25,type=float, help='threshold for filtering outliers')
    # parm for train
    parser.add_argument('--seed_list', type=int, default=[1024], help='random seed for reproduce results')
    parser.add_argument('--gpus', type=str, default='0', help='cuda ID')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--resume', type=bool, default=False, help='remember to change the root in line_222')
    parser.add_argument('--ckpt_path', type=str, default='./Noise_ckpt',
                        help='resume model')
    parser.add_argument('--log_dir', type=str, default='./Noise_train_log', help='model save dir')
    args = parser.parse_args()
    # writer = SummaryWriter(args.log_dir)  # tensorboard

    # set the train set
    # closed
    if (args.data_train == 'C_10_O_0_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/closed_list/data_file_flip_noise_10.txt'
    elif (args.data_train == 'C_20_O_0_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/closed_list/data_file_flip_noise_20.txt'
    elif (args.data_train == 'C_30_O_0_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/closed_list/data_file_flip_noise_30.txt'
    # outlier
    elif (args.data_train == 'C_0_O_10_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/outlier_list/closed_0_data_file_outlier_10.txt'
    elif (args.data_train == 'C_0_O_20_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/outlier_list/closed_0_data_file_outlier_20.txt'
    elif (args.data_train == 'C_0_O_30_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/outlier_list/closed_0_data_file_outlier_30.txt'
    # mix
    elif (args.data_train == 'C_5_O_5_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_5_data_file_outlier_5.txt'
    elif (args.data_train == 'C_10_O_10_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_10_data_file_outlier_10.txt'
    elif (args.data_train == 'C_15_O_15_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_15_data_file_outlier_15.txt'
    elif (args.data_train == 'C_20_O_20_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_20_data_file_outlier_20.txt'
    elif (args.data_train == 'C_10_O_20_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_10_data_file_outlier_20.txt'
    elif (args.data_train == 'C_10_O_30_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_10_data_file_outlier_30.txt'
    elif (args.data_train == 'C_20_O_10_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_20_data_file_outlier_10.txt'
    elif (args.data_train == 'C_30_O_10_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/close_outlier_list/closed_30_data_file_outlier_10.txt'
    elif (args.data_train == 'C0_O0_CASIA_WF'):
        args.train_file_list = '/home/user0/public2/cjp/FR/data/clean_webface_list.txt'
    else:
        print(args.data_train, ' is not available!')

    # log_init
    save_dir = os.path.join(args.log_dir,
                            args.data_train + '_' + args.backbone.upper() + '_' + args.loss_function + '_' + datetime.now().strftime(
                                '%Y%m%d_%H%M%S'))

    save_ckpt_dir = os.path.join(args.ckpt_path,
                                 args.data_train + '_' + args.backbone.upper() + '_' + args.loss_function + '_' + str(args.feature_dim) + '_' + datetime.now().strftime(
                                     '%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)

    writer = SummaryWriter(save_dir)
    logging = init_log(save_dir)
    _print = logging.info
    _print("daya_list:" + args.train_file_list)
    for seed in args.seed_list:

        # set the seed for repeating exp
        setup_seed(seed)

        # gpu_init
        multi_gpus = False
        if len(args.gpus.split(',')) > 1:
            multi_gpus = True
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # dataset loader
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

        # train data
        trainset = CASIAWebFace(args.data_root, args.train_file_list, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  pin_memory=True, num_workers=8, drop_last=False, shuffle=True)
        # NUM_CLASS = len(trainloader.dataset.classes)
        print("Number of Training Classes: {}    Number of iters per epoch: {}".format(trainset.class_nums,
                                                                                       len(trainloader)))

        # test data
        lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame = get_val_data(
            args.data_root)

        print("=" * 60)
        print("Preparing the val data...")
        lfw_dataset = ValDataset(lfw, 'lfw')
        # cfp_ff_dataset = ValDataset(cfp_ff,'cfp_ff')
        cfp_fp_dataset = ValDataset(cfp_fp, 'cfp_fp')
        agedb_dataset = ValDataset(agedb, 'agedb')
        calfw_dataset = ValDataset(calfw, 'calfw')
        cplfw_dataset = ValDataset(cplfw, 'cplfw')

        lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                                 drop_last=False, pin_memory=True)
        cfp_fp_loader = torch.utils.data.DataLoader(cfp_fp_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=2,
                                                    drop_last=False, pin_memory=True)
        agedb_loader = torch.utils.data.DataLoader(agedb_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=2,
                                                   drop_last=False, pin_memory=True)
        calfw_loader = torch.utils.data.DataLoader(calfw_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=2,
                                                   drop_last=False, pin_memory=True)
        cplfw_loader = torch.utils.data.DataLoader(cplfw_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=2,
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
        print("{} Backbone Generated".format(args.backbone))
        print("=" * 60)

        # loss_function & Hard-Mining function
        loss_function = ArcFace(args.feature_dim, trainset.class_nums)
        print("{} loss Generated".format(args.loss_function))
        print("=" * 60)

        # define optimizers for different layer
        optimizer = optim.SGD([
            {'params': net.parameters(), 'weight_decay': 5e-4},
            {'params': loss_function.parameters(), 'weight_decay': 5e-4}
        ], lr=args.lr, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[9, 18, 26], gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # set gpus
        if multi_gpus:
            net = DataParallel(net).to(device)
            loss_function = DataParallel(loss_function).to(device)

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
        thresh_value_sim = -1.0
        thresh_value_dis = 2.0


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
            # Train model
            _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epochs))
            if(args.epsilon > 0):
                _print('Epsilon:{}'.format(args.epsilon))

            net.train()
            total, correct = 0, 0
            since = time.time()
            if(epoch <5):
                thresh_value_sim = -1.0


            for data in trainloader:
                img, label = data[0].to(device), data[1].to(device)
                img_path = data[2]
                optimizer.zero_grad()
                raw_logits = net(img)
                # output = loss_function(raw_logits, label)
                # total_loss = criterion(output, label)
                # total_loss.backward()
                # optimizer.step()
                output, score = loss_function(raw_logits, label)  # cos_theta, gt
                score = score[score != -2]
                clean_output = output[score > thresh_value_sim]
                # save the noise data_path
                with open(os.path.join(save_dir, 'savedPath_' + str(epoch) + '.txt'), 'a') as f:
                    selected_indices = [i for i in range(len(score)) if score[i] <= thresh_value_sim]
                    out_path_list = [img_path[i] for i in selected_indices]
                    for i in out_path_list:
                        f.write(i + '\n')
                label_ = label[score > thresh_value_sim]
                total_loss = criterion(clean_output, label_)
                total_loss.backward()
                optimizer.step()

                # save train acc per class
                _, predict = output.max(1)
                correct += predict.eq(label).sum().item()
                total += len(label)

                total_iters += 1
                writer.add_scalar("Training_Loss", total_loss, total_iters)

                # print train information
                if total_iters % 500 == 0:
                    acc = correct / total
                    time_cur = (time.time() - since) / 500
                    since = time.time()
                    _print(
                        "Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(
                            total_iters, epoch, total_loss.item(),
                            acc * 100, time_cur, exp_lr_scheduler.get_last_lr()[0]))
            exp_lr_scheduler.step()

            # use Tensorboard to plot the curve
            # writer.add_scalar("Training_Loss", total_loss, epoch)
            # writer.add_scalar("Training_Accuracy", acc * 100, epoch)
            # update the thresh_value-------------------------------------------------------------------------------------------
            if(epoch >=2):
                _print("updating the thresh_value in epoch:{}".format(epoch))
                similarities = []
                with torch.no_grad():
                    for batch_idx, (image, label, lmdb_keys) in enumerate(trainloader):
                        image = image.to(device)
                        label = label.to(device)
                        # label = label.squeeze()
                        feat = net(image)

                        _, score = loss_function(feat, label)
                        score_id = score.cpu().numpy()
                        score_id = score_id[score_id != -2]
                        # save_scores = np.where(score_id >= thresh_value)[0].tolist()
                        similarities.extend(score_id.tolist())
                    savemat(os.path.join(save_dir, 'save_Cosine_value_in_epoch{}.mat'.format(epoch)),
                            {'Cosine': similarities})
                    cosine_distances = 1 - np.array(similarities)
                    len_outliner = len(cosine_distances[cosine_distances>thresh_value_dis])
                    # 计算四分位差
                    IQR = np.abs(np.percentile(cosine_distances, 75) - np.percentile(cosine_distances, 25))
                    # 计算标准差
                    std_dev = np.std(cosine_distances)
                    # 计算 sigma
                    sigma = 1.06 * min(std_dev, IQR / 1.34) * len(cosine_distances) ** (-0.2)
                    # 计算 vec_w
                    laplace_value = np.exp(-np.abs(cosine_distances) / sigma)
                    # 获取 outlier
                    outlier = cosine_distances[laplace_value < args.epsilon]
                    # 获取阈值
                    thresh_value_dis = np.min(outlier)
                    thresh_value_sim = 1 - thresh_value_dis

                    # thresh_value = thresh_two_peaks(similarities, epoch - 1, args.total_epochs, 0, args.milestones)
                    _print('length of outliner:%d'%(len_outliner))
                    _print('The nosiy thresh value %3f(cosineDistance); %3f(cosine_similarity)' % (
                        thresh_value_dis, thresh_value_sim))
            if epoch > int(args.total_epochs / 2):
                print("Perform Evaluation on LFW, CFP_FP, AgeDB, CALFW, CPLFW , and Save Checkpoints...")
                # LFW
                # accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(multi_gpus, device, args.feature_dim,
                #                                                               args.batch_size,
                #                                                               net, lfw, lfw_issame)
                accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val_(multi_gpus, device, args.feature_dim,
                                                                               net, lfw_loader, lfw_issame)
                buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch)
                _print('LFW Ave Accuracy: {:.4f}'.format(accuracy_lfw * 100))
                if best_lfw_acc <= accuracy_lfw * 100:
                    best_lfw_acc = accuracy_lfw * 100
                    best_lfw_epoch = epoch
                # CFP_FP
                # accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(multi_gpus, device, args.feature_dim,
                #                                                                        args.batch_size, net, cfp_fp,
                #                                                                        cfp_fp_issame)
                accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val_(multi_gpus, device,
                                                                                        args.feature_dim,
                                                                                        net, cfp_fp_loader,
                                                                                        cfp_fp_issame)
                buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch)
                _print('CFP_FP Ave Accuracy: {:.4f}'.format(accuracy_cfp_fp * 100))
                if best_cfp_fp_acc <= accuracy_cfp_fp * 100:
                    best_cfp_fp_acc = accuracy_cfp_fp * 100
                    best_cfp_fp_epoch = epoch
                # AGE_DB
                # accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(multi_gpus, device, args.feature_dim,
                #                                                                     args.batch_size, net, agedb,
                #                                                                     agedb_issame)
                accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val_(multi_gpus, device,
                                                                                     args.feature_dim, net,
                                                                                     agedb_loader,
                                                                                     agedb_issame)
                buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch)
                _print('AgeDB Ave Accuracy: {:.4f}'.format(accuracy_agedb * 100))
                if best_agedb_acc <= accuracy_agedb * 100:
                    best_agedb_acc = accuracy_agedb * 100
                    best_agedb_epoch = epoch
                # CALFW
                # accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(multi_gpus, device, args.feature_dim,
                #                                                                     args.batch_size, net, calfw,
                #                                                                     calfw_issame)
                accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val_(multi_gpus, device,
                                                                                     args.feature_dim, net,
                                                                                     calfw_loader,
                                                                                     calfw_issame)
                buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch)
                _print('CALFW Ave Accuracy: {:.4f}'.format(accuracy_calfw * 100))
                if best_calfw_acc <= accuracy_calfw * 100:
                    best_calfw_acc = accuracy_calfw * 100
                    best_calfw_epoch = epoch
                # CPLFW
                # accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(multi_gpus, device, args.feature_dim,
                #                                                                     args.batch_size, net, cplfw,
                #                                                                     cplfw_issame)
                accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val_(multi_gpus, device,
                                                                                     args.feature_dim, net,
                                                                                     cplfw_loader,
                                                                                     cplfw_issame)
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
                        loss_state_dict = loss_function.module.state_dict()
                        if not os.path.exists(save_ckpt_dir):
                            os.mkdir(save_ckpt_dir)
                        torch.save(net_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                "BatchSize{}_Seed{}_net.pth".format(args.batch_size, seed)))
                        torch.save(loss_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_loss.pth'.format(args.batch_size, seed)))
                        torch.save(checkpoint, os.path.join(save_ckpt_dir,
                                                            'BatchSize{}_Seed{}_checkpoint.pth'.format(args.batch_size,
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
                                                'BatchSize{}_Seed{}_net.pth'.format(args.batch_size, seed)))
                        torch.save(loss_state_dict,
                                   os.path.join(save_ckpt_dir,
                                                'BatchSize{}_Seed{}_loss.pth'.format(args.batch_size, seed)))
                        torch.save(checkpoint, os.path.join(save_ckpt_dir,
                                                            'BatchSize{}_Seed{}_checkpoint.pth'.format(args.batch_size,
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

                net.train()
                # save ckpt
                # if epoch % args.save_freq == 0:
                #     msg = 'Saving checkpoint: {}'.format(epoch)
                #     _print(msg)
                #     if multi_gpus:
                #         checkpoint = {
                #             'optimizer': optimizer.state_dict(),
                #             "epoch": epoch,
                #             'lr_schedule': exp_lr_scheduler.state_dict()
                #         }
                #         net_state_dict = net.module.state_dict()
                #         loss_state_dict = loss_function.module.state_dict()
                #         if not os.path.exists(save_ckpt_dir):
                #             os.mkdir(save_ckpt_dir)
                #         torch.save(net_state_dict,
                #                    os.path.join(save_ckpt_dir,
                #                                 "BatchSize{}_Seed{}_net.pth".format(args.batch_size, seed)))
                #         torch.save(loss_state_dict,
                #                    os.path.join(save_ckpt_dir,
                #                                 'BatchSize{}_Seed{}_loss.pth'.format(args.batch_size, seed)))
                #         torch.save(checkpoint, os.path.join(save_ckpt_dir,
                #                                             'BatchSize{}_Seed{}_checkpoint.pth'.format(args.batch_size,
                #                                                                                        seed)))
                #     else:
                #         checkpoint = {
                #             'optimizer': optimizer.state_dict(),
                #             "epoch": epoch,
                #             'lr_schedule': exp_lr_scheduler.state_dict()
                #         }
                #         net_state_dict = net.state_dict()
                #         loss_state_dict = loss_function.state_dict()
                #         if not os.path.exists(save_ckpt_dir):
                #             os.mkdir(save_ckpt_dir)
                #         torch.save(net_state_dict,
                #                    os.path.join(save_ckpt_dir,
                #                                 'BatchSize{}_Seed{}_net.pth'.format(args.batch_size, seed)))
                #         torch.save(loss_state_dict,
                #                    os.path.join(save_ckpt_dir,
                #                                 'BatchSize{}_Seed{}_loss.pth'.format(args.batch_size, seed)))
                #         torch.save(checkpoint, os.path.join(save_ckpt_dir,
                #                                             'BatchSize{}_Seed{}_checkpoint.pth'.format(args.batch_size,
                #                                                                                        seed)))

        print('Finish training'+'_'+str(seed))
