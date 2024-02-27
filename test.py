import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import utils.metrics as metrics
from medpy import metric
from hausdorff import hausdorff_distance
import torch.nn.functional as F
from thop import profile
import random
import time
import utils.globalvar
import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#  ============================== add the seed to make sure the results are reproducible ==============================

seed_value = 5000   # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)     # set random seed for CPU
torch.cuda.manual_seed(seed_value)      # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)   # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution

#  ================================================ parameters setting ================================================

parser = argparse.ArgumentParser(description='DTMFormer')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=4000, type=int, metavar='N', help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--dataset', default='../../dataset/ACDC/', type=str)
parser.add_argument('--modelname', default='DTMFormer', type=str, help='type of model')
parser.add_argument('--classes', type=int, default=4, help='number of classes')
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='yes', type=str)
parser.add_argument('--tensorboard', default='./tensorboard/', type=str)
parser.add_argument('--loaddirec', default="/home/wzh/code/SETR-pytorch/checkpoints_ACDC/Setr_DTMFormer.pth", type=str)
parser.add_argument('--eval_mode', default='patient', type=str)
parser.add_argument('--visual', default=False, type=bool)
parser.add_argument('--direc', default='./medt', type=str, help='directory to save')


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
val_dataset = ImageToImage2D(args.dataset, 'testofficial', tf_val, args.classes)  # no flip, return image, mask, and filename
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda")

from models import SETR
model = SETR.Setr_DTMFormer(classes=args.classes)

# model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load(args.loaddirec))
model.eval()

input = torch.randn(1, 1, 256, 256).cuda()
flops, params = profile(model, inputs=(input, ))
print("GFLOPS: {}".format(flops/1e9))
print("参数量: {}M".format(params/1e6))

#  ============================================= begin to eval the model =============================================
dices = 0
hds = 0
ious = 0
ses = 0
sps = 0
accs, fs, ps, rs = 0, 0, 0, 0
times = 0
smooth = 1e-25
mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
mses, msps, mious = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)
maccs, mfs, mps, mrs = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)

if eval_mode == "slice":
    for batch_idx, (imgs, mask, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        test_img_path = os.path.join(args.dataset + '/img', image_filename)
        from utils.imgname import keep_img_name

        keep_img_name(test_img_path)

        imgs = Variable(imgs.to(device='cuda'))
        mask = Variable(mask.to(device='cuda'))

        with torch.no_grad():
            outputs, _ = model(imgs)
        y_out = F.softmax(outputs, dim=1)


        gt = mask.detach().cpu().numpy()
        pred = y_out.detach().cpu().numpy()
        seg = np.argmax(pred, axis=1)
        b, h, w = seg.shape
        for i in range(1, args.classes):
            pred_i = np.zeros((b, h, w))
            pred_i[seg == i] = 255
            gt_i = np.zeros((b, h, w))
            gt_i[gt == i] = 255
            mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
            mhds[i] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
            se, sp, iou, acc, f, precision, recall = metrics.sespiou_coefficient2(pred_i, gt_i)
            maccs[i] += acc
            mfs[i] += f
            mps[i] += precision
            mrs[i] += recall

            mses[i] += se
            msps[i] += sp
            mious[i] += iou
            del pred_i, gt_i

        if args.visual:
            import cv2
            img_ori = cv2.imread(os.path.join(args.dataset + '/img', image_filename))
            img = np.zeros((h, w, 3))
            img_r = img_ori[:, :, 0]
            img_g = img_ori[:, :, 1]
            img_b = img_ori[:, :, 2]
            table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
            seg0 = seg[0, :, :]

            for i in range(1, args.classes):
                img_r[seg0 == i] = table[i - 1, 0]
                img_g[seg0 == i] = table[i - 1, 1]
                img_b[seg0 == i] = table[i - 1, 2]

            img[:, :, 0] = img_r
            img[:, :, 1] = img_g
            img[:, :, 2] = img_b
            img = np.uint8(img)

            dice_pic = np.zeros(args.classes)
            for i in range(1, args.classes):
                pred_i = np.zeros((b, h, w))
                pred_i[seg == i] = 255
                gt_i = np.zeros((b, h, w))
                gt_i[gt == i] = 255
                dice_pic[i] = metrics.dice_coefficient(pred_i, gt_i)
            dice = dice_pic.sum() / (args.classes - 1)
            num = "%06d" % utils.globalvar.CNT

            fulldir = args.direc + "/"
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)
            image_filename = num + '_' + image_filename.replace('.png', '') + '_' + str(dice) + '.png'
            cv2.imwrite(fulldir + image_filename, img)
        utils.globalvar.CNT += 1

    mdices = mdices / (batch_idx + 1)
    mhds = mhds / (batch_idx + 1)
    mses = mses / (batch_idx + 1)
    msps = msps / (batch_idx + 1)
    mious = mious / (batch_idx + 1)

    maccs = maccs / (batch_idx + 1)
    mfs = mfs / (batch_idx + 1)
    mps = mps / (batch_idx + 1)
    mrs = mrs / (batch_idx + 1)

    for i in range(1, args.classes):
        dices += mdices[i]
        hds += mhds[i]
        ses += mses[i]
        sps += msps[i]
        ious += mious[i]

        accs += maccs[i]
        fs += mfs[i]
        ps += mps[i]
        rs += mrs[i]
    print(mdices, '\n', mhds, '\n', mses, '\n', msps, '\n', mious, '\n')
    print(dices / (args.classes - 1), hds / (args.classes - 1), ses / (args.classes - 1), sps / (args.classes - 1),
          ious / (args.classes - 1))
    print(accs / (args.classes - 1), fs / (args.classes - 1), ps / (args.classes - 1), rs / (args.classes - 1))
    print(times)

else:

    flag = np.zeros(2000)
    times = 0
    mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
    mses, msps, mious = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)
    current_time = time.time()

    for batch_idx, (imgs, mask, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        imgs = Variable(imgs.to(device='cuda'))
        mask = Variable(mask.to(device='cuda'))

        with torch.no_grad():
            # outputs = model(imgs)
            outputs, _ = model(imgs)
        y_out = F.softmax(outputs, dim=1)

        gt = mask.detach().cpu().numpy()
        pred = y_out.detach().cpu().numpy()  
        seg = np.argmax(pred, axis=1)

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
                    if pred_i.sum() != 0 and gt_i.sum() != 0:
                        mhds[i] += metric.binary.hd95(pred_i, gt_i)
                    se, sp, iou, acc, f, precision, recall = metrics.sespiou_coefficient2(pred_i, gt_i)
                    maccs[i] += acc
                    mfs[i] += f
                    mps[i] += precision
                    mrs[i] += recall
                    mses[i] += se
                    msps[i] += sp
                    mious[i] += iou
                    del pred_i, gt_i
            seg_patient = seg[:, None, :, :]
            gt_patient = gt[:, None, :, :]
            flag[patientid] = 1
        else:
            seg_patient = np.concatenate((seg_patient, seg[:, None, :, :]), axis=1)
            gt_patient = np.concatenate((gt_patient, gt[:, None, :, :]), axis=1)

        utils.globalvar.CNT += 1
        # ---------------the last patient--------------
    b, s, h, w = seg_patient.shape
    for i in range(1, args.classes):
        pred_i = np.zeros((b, s, h, w))
        pred_i[seg_patient == i] = 1
        gt_i = np.zeros((b, s, h, w))
        gt_i[gt_patient == i] = 1
        mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
        if pred_i.sum() != 0 and gt_i.sum() != 0:
            mhds[i] += metric.binary.hd95(pred_i, gt_i)
        se, sp, iou, acc, f, precision, recall = metrics.sespiou_coefficient2(pred_i, gt_i)
        maccs[i] += acc
        mfs[i] += f
        mps[i] += precision
        mrs[i] += recall

        mses[i] += se
        msps[i] += sp
        mious[i] += iou
        del pred_i, gt_i
    patients = np.sum(flag)
    mdices, mhds, mses, msps, mious = mdices / patients, mhds / patients, mses / patients, msps / patients, mious / patients
    maccs, mfs, mps, mrs = maccs / patients, mfs / patients, mps / patients, mrs / patients

    print("--------------------------------")
    print("mdices, mhds, mses, msps, mious")
    print(mdices)
    print(mhds)
    print(mses)
    print(msps)
    print(mious)

    print("--------------------------------")
    print("maccs, mfs, mps, mrs")
    print(maccs)
    print(mfs)
    print(mps)
    print(mrs)
    print("--------------------------------")
    for i in range(1, args.classes):
        dices += mdices[i]
        hds += mhds[i]
        ious += mious[i]
        ses += mses[i]
        sps += msps[i]

        accs += maccs[i]
        fs += mfs[i]
        ps += mps[i]
        rs += mrs[i]

    print("dices, hds, ious, ses, sps")
    print(dices / (args.classes - 1))
    print(hds / (args.classes - 1))
    print(ious / (args.classes - 1))
    print(ses / (args.classes - 1))
    print(sps / (args.classes - 1))
    print("accs, fs, ps, rs")
    print(accs / (args.classes - 1))
    print(fs / (args.classes - 1))
    print(ps / (args.classes - 1))
    print(rs / (args.classes - 1))

    print("val time: %g" % (time.time() - current_time))


