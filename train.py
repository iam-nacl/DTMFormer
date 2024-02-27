import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import utils.metrics as metrics
import time
import torch.nn.functional as F
import random
from utils.lr import *
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from einops import rearrange
from thop import profile
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#  ============================== add the seed to make sure the results are reproducible ==============================

seed_value = 5000  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution

#  ================================================ parameters setting ================================================

parser = argparse.ArgumentParser(description='DTMFormer')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='batch size (default: 4)')
parser.add_argument('--learning_rate', default=5e-4, type=float, metavar='LR', help='更改后 learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--dataset', default='../../dataset/ACDC/', type=str)
parser.add_argument('--modelname', default='SETR_PUP', type=str, help='type of model')
parser.add_argument('--classes', type=int, default=4, help='number of classes')
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--direc', default='./medt', type=str, help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--patchsize', type=int, default=8)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='yes', type=str)
parser.add_argument('--tensorboard', default='./tensorboard_ACDC/', type=str)
parser.add_argument('--eval_mode', default='patient', type=str)

#  =============================================== model initialization ===============================================

args = parser.parse_args()
direc = args.direc  # the path of saving model
eval_mode = args.eval_mode

if args.gray == "yes":
    from utils.utils_multi import JointTransform2D, ImageToImage2D
    imgchant = 1
else:
    from utils.utils_multi_rgb import JointTransform2D, ImageToImage2D
    imgchant = 3
if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None


tf_train = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0, p_contr=0.0, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
tf_val = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_gama=0, color_jitter_params=None, long_mask=True)  # image reprocessing
train_dataset = ImageToImage2D(args.dataset, 'trainofficial', tf_train, args.classes)  # only random horizontal flip, return image, mask, and filename
val_dataset = ImageToImage2D(args.dataset, 'valofficial', tf_val, args.classes)  # no flip, return image, mask, and filename
test_dataset = ImageToImage2D(args.dataset, 'testofficial', tf_val, args.classes)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device("cuda")

from models import SETR
model = SETR.Setr_DTMFormer(classes=args.classes)
# model = nn.DataParallel(model)
model.to(device)

loss_fn = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
loss_SmoothL1 = nn.SmoothL1Loss()

input = torch.randn(1, 1, 256, 256).cuda()
flops, params = profile(model, inputs=(input, ))
print("GFLOPS: {}".format(flops/1e9))
print("参数量: {}M".format(params/1e6))

timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
boardpath = args.tensorboard + timestr + '_' + args.modelname
if not os.path.isdir(boardpath):
    os.makedirs(boardpath)
TensorWriter = SummaryWriter(boardpath)

#  ============================================= begin to train the model =============================================
best_dice = 0.0
epoch_losses_train = []
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=1e-5)

for epoch in range(args.epochs):
    #  ---------------------------------- training ----------------------------------
    model.train()
    batch_losses = []
    batch_losses1 = []
    batch_losses2 = []
    current_time = time.time()
    for step, (imgs, label_imgs, img_ids) in enumerate(dataloader):

        imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))
        outputs, outputs_as = model(imgs)

        label_as = rearrange(label_imgs, 'b (h p1) (w p2) -> b h w (p1 p2)', h=int(args.imgsize/args.patchsize), w=int(args.imgsize/args.patchsize), p1=args.patchsize, p2=args.patchsize).max(dim=-1).values
        label_as = (label_as != 0).float()

        # compute the loss:
        loss1 = loss_fn(outputs, label_imgs)
        loss2 = loss_SmoothL1(outputs_as, label_as)
        loss = loss1 + 0.3 * loss2

        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        loss_value1 = loss1.data.cpu().numpy()
        loss_value2 = loss2.data.cpu().numpy()
        batch_losses1.append(loss_value1)
        batch_losses2.append(loss_value2)

        # optimization step:
        optimizer.zero_grad()  # (reset gradients)
        loss.backward()  # (compute gradients)
        optimizer.step()  # (perform optimization step)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)

    epoch_loss1 = np.mean(batch_losses1)
    epoch_loss2 = np.mean(batch_losses2)
    print("train loss: %g" % epoch_loss)
    print("train time: %g" % (time.time() - current_time))
    TensorWriter.add_scalar('train_loss', epoch_loss, epoch)
    TensorWriter.add_scalar('loss1', epoch_loss1, epoch)
    TensorWriter.add_scalar('loss2', epoch_loss2, epoch)

    #  ----------------------------------- evaluate -----------------------------------
    val_loss = 0
    dices = 0
    hds = 0
    smooth = 1e-25
    mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
    flag = np.zeros(200)  # record the patients

    model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []

    current_time = time.time()
    for batch_idx, (imgs, mask, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        imgs = Variable(imgs.to(device='cuda'))
        mask = Variable(mask.to(device='cuda'))

        # 源代码------------------------
        # start = timeit.default_timer()
        with torch.no_grad():
            outputs,_ = model(imgs)
        # stop = timeit.default_timer()
        # print('Time: ', stop - start)
        outputs = F.softmax(outputs, dim=1)
        # val_loss = smoothl1(outputs, mask)
        loss = loss_fn(outputs, mask)
        val_loss = loss.data.cpu().numpy()
        # 源代码------------------------

        gt = mask.detach().cpu().numpy()
        pred = outputs.detach().cpu().numpy()  # (b, c,h, w) tep
        seg = np.argmax(pred, axis=1)  # (b, h, w) whether exist same score?

        patientid = int(image_filename[:3])
        if flag[patientid] == 0:
            if np.sum(flag) > 0:  # compute the former result
                b, s, h, w = seg_patient.shape
                for i in range(1, args.classes):
                    pred_i = np.zeros((b, s, h, w))
                    pred_i[seg_patient == i] = 1
                    gt_i = np.zeros((b, s, h, w))
                    gt_i[gt_patient == i] = 1
                    mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                    del pred_i, gt_i
            seg_patient = seg[:, None, :, :]
            gt_patient = gt[:, None, :, :]
            flag[patientid] = 1
        else:
            seg_patient = np.concatenate((seg_patient, seg[:, None, :, :]), axis=1)
            gt_patient = np.concatenate((gt_patient, gt[:, None, :, :]), axis=1)
        # ---------------the last patient--------------
    b, s, h, w = seg_patient.shape
    for i in range(1, args.classes):
        pred_i = np.zeros((b, s, h, w))
        pred_i[seg_patient == i] = 1
        gt_i = np.zeros((b, s, h, w))
        gt_i[gt_patient == i] = 1
        mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
        del pred_i, gt_i
    patients = np.sum(flag)
    mdices = mdices / patients
    for i in range(1, args.classes):
        dices += mdices[i]
    print('epoch [{}/{}], test loss:{:.4f}'.format(epoch, args.epochs, val_loss / (batch_idx + 1)))
    print('epoch [{}/{}], test dice:{:.4f}'.format(epoch, args.epochs, dices / (args.classes - 1)))
    print("val time: %g" % (time.time() - current_time))
    TensorWriter.add_scalar('val_loss', val_loss / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('dices', dices / (args.classes - 1), epoch)
    if dices / (args.classes - 1) > best_dice or epoch == args.epochs - 1:
        best_dice = dices / (args.classes - 1)
        timestr = time.strftime('%m%d%H%M')
        save_path = './checkpoints_ACDC/' + '%s_' % timestr + args.modelname + '_%s' % epoch + '_' + str(best_dice)
        torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

    print("lr: ", optimizer.param_groups[0]['lr'])
    TensorWriter.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    optimizer.param_groups[0]['lr'] = poly_lr(epoch, args.epochs - 1, args.learning_rate, 0.9)
