import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
VALDATA_LIST_PATH = './dataset/gta5_list/val.txt'
IGNORE_LABEL = 255
#INPUT_SIZE = '1280,720'
INPUT_SIZE = '512,512'
DATA_DIRECTORY_TARGET = './data/cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
VALDATA_LIST_PATH_TARGET = './dataset/cityscapes_list/val.txt'
INPUT_SIZE_TARGET = '512,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
#NUM_CLASSES = 19
NUM_CLASSES = 20
NUM_STEPS = 250000
#NUM_STEPS = 5000
#NUM_STEPS_STOP = 80000  # early stopping
NUM_STEPS_STOP = 5000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-3
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--valdata-list", type=str, default=VALDATA_LIST_PATH,
                        help="Path to the file listing the images in the source val dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--valdata-list-target", type=str, default=VALDATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target val dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    #cudnn.enabled = True
    gpu = args.gpu

    # Create network
    #if args.model == 'DeepLab':
    #    model = DeeplabMulti(num_classes=args.num_classes)
    #    if args.restore_from[:4] == 'http' :
    #        saved_state_dict = model_zoo.load_url(args.restore_from)
    #    else:
    #        saved_state_dict = torch.load(args.restore_from)

    #    new_params = model.state_dict().copy()
    #    for i in saved_state_dict:
    #        # Scale.layer5.conv2d_list.3.weight
    #        i_parts = i.split('.')
    #        # print i_parts
    #        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
    #            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
    #    model.load_state_dict(new_params)

    #model.train()
    #model.cuda(args.gpu)

    #cudnn.benchmark = True

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    model_D.train()
    model_D.cuda(0)
    #model_D1 = FCDiscriminator(num_classes=args.num_classes)
    #model_D2 = FCDiscriminator(num_classes=args.num_classes)

    #model_D1.train()
    #model_D1.cuda(args.gpu)

    #model_D2.train()
    #model_D2.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    gta_trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, args.num_classes, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    gta_trainloader_iter = enumerate(gta_trainloader)


    gta_valloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.valdata_list, args.num_classes, max_iters=None,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    gta_valloader_iter = enumerate(gta_valloader)


    cityscapes_targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target, args.num_classes,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set='train'),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    cityscapes_targetloader_iter = enumerate(cityscapes_targetloader)

    cityscapes_valtargetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.valdata_list_target, args.num_classes,
                                                     max_iters=None,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set='val'),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    cityscapes_valtargetloader_iter = enumerate(cityscapes_valtargetloader)



    # implement model.optim_parameters(args) to handle different models' lr setting

    #optimizer = optim.SGD(model.optim_parameters(args),
    #                      lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    #optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    #optimizer_D1.zero_grad()

    #optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    #optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    #interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_D_value = 0
        #loss_seg_value1 = 0
        #loss_adv_target_value1 = 0
        #loss_D_value1 = 0

        #loss_seg_value2 = 0
        #loss_adv_target_value2 = 0
        #loss_D_value2 = 0

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        _, batch = gta_trainloader_iter.next()
        images, labels, _, _ = batch
        size = labels.size()
        #print(size)
        #labels = Variable(labels)
        oneHot_size = (size[0], args.num_classes, size[2], size[3])
        input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, labels.long(), 1.0)
        #print(input_label.size())
        labels1 = Variable(input_label).cuda(0)
        #D_out1 = model_D(labels)

        #print(D_out1.data.size())
        #loss_out1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(0))

        _, batch = cityscapes_targetloader_iter.next()
        images, labels, _, _ = batch
        size = labels.size()
        #labels = Variable(labels)
        oneHot_size = (size[0], args.num_classes, size[2], size[3])
        input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()

        input_label = input_label.scatter_(1, labels.long(), 1.0)
        labels2 = Variable(input_label).cuda(0)

        #print(labels1.data.size())
        #print(labels2.data.size())
        labels = torch.cat((labels1, labels2), 0)
        #print(labels.data.size())


        #D_out2 = model_D(labels)
        D_out = model_D(labels)
        #print(D_out.data.size())

        target_size = D_out.data.size()
        target_labels1 = torch.FloatTensor(torch.Size((target_size[0]/2, target_size[1], target_size[2], target_size[3]))).fill_(source_label)
        target_labels2 = torch.FloatTensor(torch.Size((target_size[0]/2, target_size[1], target_size[2], target_size[3]))).fill_(target_label)
        target_labels = torch.cat((target_labels1, target_labels2), 0)
        target_labels = Variable(target_labels).cuda(0)
        #print(target_labels.data.size())
        loss_out = bce_loss(D_out, target_labels)

        loss = loss_out / args.iter_size
        loss.backward()

        loss_D_value += loss_out.data.cpu().numpy()[0] / args.iter_size

        #print(loss_D_value)
        optimizer_D.step()

        #print('exp = {}'.format(args.snapshot_dir))
        if i_iter %100 == 0:
            print('iter = {0:8d}/{1:8d}, loss_D = {2:.3f}'.format(i_iter, args.num_steps, loss_D_value))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'Classify_' + str(args.num_steps) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'Classify_' + str(i_iter) + '.pth'))

            model_D.eval()
            loss_valD_value = 0
            correct = 0
            wrong = 0
            for i, (images, labels, _, _) in enumerate(gta_valloader):
                #if i > 500:
                #    break
                size = labels.size()
                #labels = Variable(labels)
                oneHot_size = (size[0], args.num_classes, size[2], size[3])
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, labels.long(), 1.0)
                labels = Variable(input_label).cuda(0)
                D_out1 = model_D(labels)
                loss_out1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(0))
                loss_valD_value += loss_out1.data.cpu().numpy()[0]
                correct = correct + (D_out1.data.cpu() < 0).sum()/100
                wrong = wrong + (D_out1.data.cpu() >=0).sum()/100
                #accuracy = 1.0 * correct / (wrong + correct)
                #print('accuracy:%f' % accuracy)
                #print(correct)
                #print(wrong)


            for i, (images, labels, _, _) in enumerate(cityscapes_valtargetloader):
                #if i > 500:
                #    break
                size = labels.size()
                #labels = Variable(labels)
                oneHot_size = (size[0], args.num_classes, size[2], size[3])
                input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_label = input_label.scatter_(1, labels.long(), 1.0)
                labels = Variable(input_label).cuda(0)
                D_out2 = model_D(labels)
                loss_out2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(0))
                loss_valD_value += loss_out2.data.cpu().numpy()[0]
                wrong = wrong + (D_out2.data.cpu() < 0).sum()
                correct = correct + (D_out2.data.cpu() >=0).sum()
            accuracy = 1.0 * correct / (wrong + correct)
            print('accuracy:%f' % accuracy)

            print('iter = {0:8d}/{1:8d}, loss_valD = {2:.3f}'.format(i_iter, args.num_steps, loss_valD_value))


            model_D.train()





if __name__ == '__main__':
    main()
