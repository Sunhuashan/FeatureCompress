import argparse
import logging
from fvc import *
from fvc_net.ms_ssim_torch import *

from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import json

from typing import List
from fvc import FVC_base
from plt.hevc_draw_points import hevccdrawplt
torch.backends.cudnn.enabled = True

from fvc_net.RDloss import *
from dataset import HEVC_ClassCDataSet
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0#  // gpu_num
gpu_per_batch = 4
test_step = 10000#  // gpu_num
tot_epoch = 1000000
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)

parser = argparse.ArgumentParser(description='FVC reimplement')

parser.add_argument('-l', '--log', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--testhevcC', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', required=True,
        help = 'hyperparameter of Reid in json format')

def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        # ref_i_dir = geti(train_lambda)
        ref_i_dir = get_mbt2018_i(train_lambda)


    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']




def testhevcC(global_step, testfull=False):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumbpprate = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d" % (batch_idx, len(test_loader)))
            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            framerate = input[5].cuda()
            shape = input[6]

            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
            sumbpprate += torch.mean(ref_bpp).detach() * framerate * shape[0].cuda() * shape[1].cuda()
            sumpsnr += torch.mean(ref_psnr).detach()
            summsssim += torch.mean(ref_msssim).detach()
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]

                torch.use_deterministic_algorithms(True)
                torch.set_num_threads(1)

                bpp, reconframe = net.test(input_image, ref_image)

                torch.use_deterministic_algorithms(False)
                torch.set_num_threads(36)

                mse_loss = torch.mean((reconframe - input_image).pow(2))

                sumbpp += torch.mean(bpp).cuda().detach()
                sumbpprate += torch.mean(bpp).detach() * framerate * shape[0].cuda() * shape[1].cuda()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,size_average=True)
                cnt += 1
                ref_image = reconframe
        log = "global step :%d " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumbpprate /= (cnt * 1024)
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVC_ClassC dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf , average bpprate:%.6lf\n" % (sumbpp, sumpsnr, summsssim, sumbpprate)
        logger.info(log)
        sumbpp = sumbpp.cpu().numpy()
        sumpsnr = sumpsnr.cpu().numpy()
        summsssim = summsssim.cpu().numpy()
        hevccdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(seed=0)


    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.log != '':
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("FVC training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)

    model = FVC_base()
    net = model.cuda()

    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)

    if args.testhevcC:
        net.update(force=True)
        test_dataset = HEVC_ClassCDataSet(refdir=ref_i_dir, testfull=True)
        print('testing HEVC_ClassC')
        testhevcC(global_step, testfull=True)
        exit(0)




