import argparse
import logging
from fvc import *
from fvc_net.ms_ssim_torch import *
from fvc import FVC_base
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import json

from typing import List
from fvc import FVC_base
from dataset import DataSet, UVGDataSet
from tensorboardX import SummaryWriter
torch.backends.cudnn.enabled = True
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


from dataset import *



logger = logging.getLogger("VideoCompression")


def testhevcA(global_step,net,ref_i_dir,testfull=False):
    net.update(force=True)
    # print('testing HEVC_ClassA')
    test_dataset = HEVC_ClassADataSet(refdir=ref_i_dir, testfull=True)

    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=2, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0

        for batch_idx, input in enumerate(test_loader):
            # if batch_idx % 10 == 0:
            #     print("testing : %d/%d"% (batch_idx, len(test_loader)))
            if batch_idx==11:
                break

            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
            sumpsnr += torch.mean(ref_psnr).detach()
            summsssim += torch.mean(ref_msssim).detach()
            cnt += 1

            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]

                torch.use_deterministic_algorithms(True)
                torch.set_num_threads(1)

                bpp,reconframe=net.test(input_image,ref_image)

                torch.use_deterministic_algorithms(False)
                torch.set_num_threads(36)

                mse_loss = torch.mean((reconframe - input_image).pow(2))

                sumbpp += torch.mean(bpp).cuda().detach()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVC_ClassA dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)

def testhevcB(global_step,net,ref_i_dir,testfull=False):
    net.update(force=True)
    # print('testing HEVC_ClassB')
    test_dataset = HEVC_ClassBDataSet(refdir=ref_i_dir, testfull=True)
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=2, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0

        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d"% (batch_idx, len(test_loader)))

            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
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
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,
                                     size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVC_ClassB dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)

def testhevcC(global_step,net,ref_i_dir,testfull=False):
    net.update(force=True)
    # print('testing HEVC_ClassC')
    test_dataset = HEVC_ClassCDataSet(refdir=ref_i_dir, testfull=True)
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            # if batch_idx % 10 == 0:
            #     print("testing : %d/%d" % (batch_idx, len(test_loader)))
            if batch_idx==11:
                break
            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
            sumpsnr += torch.mean(ref_psnr).detach()
            summsssim += torch.mean(ref_msssim).detach()
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]

                bpp,reconframe=net.test(input_image,ref_image)
                mse_loss = torch.mean((reconframe - input_image).pow(2))

                sumbpp += torch.mean(bpp).cuda().detach()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,
                                     size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVC_ClassC dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)

def testhevcD(global_step,net,ref_i_dir,testfull=False):
    net.update(force=True)
    # print('testing HEVC_ClassD')
    test_dataset = HEVC_ClassDDataSet(refdir=ref_i_dir, testfull=True)
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0

        for batch_idx, input in enumerate(test_loader):
            # if batch_idx % 10 == 0:
            #     print("testing : %d/%d"% (batch_idx, len(test_loader)))
            if batch_idx == 11:
                break
            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
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
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,
                                     size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVC_ClassD dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)

def testhevcE(global_step,net,ref_i_dir,testfull=False):
    net.update(force=True)
    # print('testing HEVC_ClassE')
    test_dataset = HEVC_ClassEDataSet(refdir=ref_i_dir, testfull=True)
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            # if batch_idx % 10 == 0:
            #     print("testing : %d/%d" % (batch_idx, len(test_loader)))
            if batch_idx==11:
                break
            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
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
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,
                                     size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "HEVC_ClassE dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)


def testmcl(global_step,net,ref_i_dir,testfull=False):
    net.update(force=True)
    # print('testing MCL_JCV')
    test_dataset = MCL_JCVDataSet(refdir=ref_i_dir, testfull=True)
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            # if batch_idx % 12 == 0:
            #     print("testing : %d/%d" % (batch_idx, len(test_loader)))
            if batch_idx==13:
                break
            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
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
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,
                                     size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "MCL_JCV dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)


def testuvg(global_step,net,ref_i_dir,testfull=False):
    net.update(force=True)
    # print('testing UVG')
    test_dataset = UVGDataSet(refdir=ref_i_dir, testfull=True)
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            # if batch_idx % 12 == 0:
            #     print("testing : %d/%d" % (batch_idx, len(test_loader)))
            if batch_idx==13:
                break
            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
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
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,
                                     size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)


def testvtl(global_step,net,ref_i_dir,testfull=False):
    # print('testing VTL')
    net.update(force=True)
    test_dataset = VTLDataSet(refdir=ref_i_dir, testfull=True)

    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            # if batch_idx % 12 == 0:
            #     print("testing : %d/%d"% (batch_idx, len(test_loader)))
            if batch_idx==13:
                break
            input_images = input[0].cuda()
            ref_image = input[1].cuda()
            ref_bpp = input[2].cuda()
            ref_psnr = input[3].cuda()
            ref_msssim = input[4].cuda()
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach()
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
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
                summsssim += ms_ssim(reconframe.cuda().detach(), input_image.cuda().detach(), data_range=1.0,
                                     size_average=True)
                cnt += 1
                ref_image = reconframe
        # log = "global step %d : " % (global_step) + "\n"
        # logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "VTLdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)

from compressai.zoo import cheng2020_anchor

def cal_rd_cost(distortion: torch.Tensor, bpp: torch.Tensor, lambda_weight: float = 1024):
    # 计算率失真代价（RD cost）
    rd_cost = lambda_weight * distortion + bpp  # RD cost = lambda_weight * distortion + bpp
    return rd_cost  # 返回 RD cost


def cal_bpp(likelihood: torch.Tensor, num_pixels: int):
    # 计算每像素比特率（bits per pixel，bpp）
    bpp = torch.log(likelihood).sum() / (-math.log(2) * num_pixels)  # bpp = log(likelihood)之和 / (-log(2) * num_pixels)
    return bpp  # 返回 bpp


def cal_distoration(A: torch.Tensor, B:torch.Tensor):
    # 计算失真度
    dis = nn.MSELoss()  # 定义均方误差损失函数
    return dis(A, B)  # 返回 A 和 B 之间的均方误差


def cal_psnr(distortion: torch.Tensor):
    # 计算峰值信噪比（PSNR）
    psnr = -10 * torch.log10(distortion)  # PSNR = -10 * log10(distortion)
    return psnr  # 返回 PSNR


def Var(x):
    # 将 x 转换为 CUDA 变量
    return Variable(x.cuda())  # 返回 CUDA 变量


def testHEVC(global_step, net, filelist, testfull=True): # 定义测试函数，接受四个参数：global_step，net，filelist 和 testfull。
    net.update(force=True)  # 更新网络
    # I_codec = cheng2020_anchor(quality=lambda_I_quality_map[args.lambda_weight], metric='mse', pretrained=True).cuda()
    I_codec = cheng2020_anchor(quality=6, metric='mse', pretrained=True).cuda()
    I_codec.eval()

    test_dataset = HEVCDataSet(filelist=filelist, testfull=testfull)  # 定义测试数据集
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1,
                                 pin_memory=True)  # 定义测试数据加载器

    sumbpp = 0  # 初始化总 bpp
    sumpsnr = 0  # 初始化总 PSNR
    summsssim = 0  # 初始化总 MS-SSIM
    cnt = 0  # 初始化计数器
    gop_num = 0

    with torch.no_grad():  # 禁用梯度计算
        
        net.eval()  # 将网络设置为评估模式
        
        for batch_idx, input in enumerate(test_loader):  # 遍历测试数据加载器中的每个批次
            # input (1, 10, C, H, W)
            if batch_idx == 11:  # 如果批次索引等于11，则跳出循环
                break

            input_images = input[0].cuda()  # 获取输入图像
            seqlen = input_images.size()[0]  # 获取序列长度/图片张数

            for i in range(seqlen):
                if i == 0:  # 如果是第一帧
                    I_frame = input_images[i, :, :, :]   # 获取 I 帧并将其转移到 GPU 上
                    # print(I_frame.shape)   # 打印 I 帧的形状
                    num_pixels = 1 * I_frame.shape[1] * I_frame.shape[2]   # 计算像素数量
                    arr = I_codec(torch.unsqueeze(I_frame, 0))   # 对 I 帧进行编码，返回一个字典，包含编码结果和概率分布等信息
                    I_rec = arr['x_hat']   # 获取重建的 I 帧
                    I_likelihood_y = arr["likelihoods"]['y']  # 获取 y 方向概率分布
                    I_likelihood_z = arr["likelihoods"]['z']  # 获取 z 方向概率分布

                    ref_image = I_rec.clone().detach()  # 获取参考图像（即重建的 I 帧）
                    y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels).cpu().detach().numpy()  # 计算 y 方向 bpp
                    z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()   # 计算 z 方向 bpp
                    bpp = y_bpp + z_bpp   # 计算总 bpp
                    psnr = cal_psnr(distortion=cal_distoration(I_rec, I_frame)).cpu().detach().numpy()   # 计算 PSNR

                    I_frame = torch.unsqueeze(I_frame, 0)
                    msssim = ms_ssim(I_rec.cpu().detach(), I_frame.cpu().detach(), data_range=1.0,
                                        size_average=True)   #  MS-SSIM

                    print("------------------ GOP {0} --------------------".format(batch_idx + 1))   # 打印 GOP 分割线和编号
                    print("I frame: ", bpp, "\t", psnr)   # 打印 I 帧的 bpp 和 PSNR

                    gop_num += 1  # GOP 数量加一

                else:
                    cur_frame = input_images[i, :, :, :].cuda()  # 获取当前帧并将其转移到 GPU 上
                    cur_frame, ref_image = Var(torch.unsqueeze(cur_frame, 0)), Var(ref_image)  # 将当前帧、参考图像和左侧参考帧转换为 CUDA 变量

                    torch.use_deterministic_algorithms(True)   # 启用确定性算法
                    torch.set_num_threads(1)   # 设置线程数为1

                    bpp, reconframe = net.test(cur_frame, ref_image)    # 使用网络进行测试，返回 bpp 和重建帧

                    torch.use_deterministic_algorithms(False)   # 禁用确定性算法
                    torch.set_num_threads(36)   # 设置线程数为36
                    mse_loss = torch.mean((reconframe - cur_frame).pow(2))   # 计算均方误差损失
                    
                    bpp = torch.mean(bpp).cpu().detach()   # 累加 bpp 的平均值
                    psnr = torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach()   # 累加 PSNR 的平均值
                    msssim = ms_ssim(reconframe.cpu().detach(), cur_frame.cpu().detach(), data_range=1.0, size_average=True)   # 累加 MS-SSIM 的平均值
                    rd_cost = cal_rd_cost(distortion=mse_loss, bpp=bpp).cpu()  # 计算 RD cost
                    
                    ref_image = reconframe   # 更新参考图像为重建帧

                cnt += 1
                sumbpp += bpp
                sumpsnr += psnr
                summsssim += msssim    

        ave_bpp = sumbpp / cnt
        ave_psnr = sumpsnr / cnt
        ave_msssim = summsssim / cnt
        log = "HEVC_ClassC dataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (ave_bpp, ave_psnr, ave_msssim)
        logger.info(log)

