import os

import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
# from subnet.basics import *
from fvc_net.ms_ssim_torch import ms_ssim
from augmentation import random_flip, random_crop_and_pad_image_and_labels
from fvc_net.basic import *

# trainpath='F:/vc_project/video_compression/data/train_data'
# trainpath='G:/wangyiming/data/train_data'
# trainpath='C:/Users/Administrator/Desktop'
trainpath = 'C:/Users/lenovo/Documents/Workspace/Data/train'

# testpath= 'G:/wangyiming/data/test_data'
# testpath= 'D:/data/test_data'
testpath = ''


class DataSet(data.Dataset):
    def __init__(self, path=trainpath + "/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_ref_list, self.image_input_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir=trainpath + "/vimeo_septuplet/sequences/",
                  filefolderlist="data/vimeo_septuplet/test.txt"):
        """
            return input frame filename list and its last frame filename list (reference frame filename list)
        """
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]

            # last frame filename
            refnumber = int(y[-5:-4]) - 1
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

        return fns_train_ref, fns_train_input

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        # convert from (height, width, channel) to (channel, height, width)
        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image,
                                                                      [self.im_height, self.im_width])

        input_image, ref_image = random_flip(input_image, ref_image)

        return input_image, ref_image

class ThreeFrameDataSet(data.Dataset):
    def __init__(self, path=trainpath + "/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_ref_list, self.image_mid_list, self.image_input_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir=trainpath + "/vimeo_septuplet/sequences/",
                  filefolderlist="data/vimeo_septuplet/test.txt"):
        """
            return input frame filename list and its last frame filename list (reference frame filename list)
        """
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []
        fns_train_mid = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]

            # last frame filename
            refnumber = int(y[-5:-4]) - 1
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_mid += [refname]

            # last last frame filename
            refnumber = int(y[-5:-4]) - 2
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

        return fns_train_ref, fns_train_mid, fns_train_input

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        mid_image = imageio.imread(self.image_mid_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        mid_image = mid_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        # convert from (height, width, channel) to (channel, height, width)
        input_image = input_image.transpose(2, 0, 1)
        mid_image = mid_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        input_image = torch.from_numpy(input_image).float()
        mid_image = torch.from_numpy(mid_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        cropped_mid_image, cropped_ref_image = random_crop_and_pad_image_and_labels(mid_image, ref_image,
                                                                      [self.im_height, self.im_width])

        cropped_input_image, cropped_ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image,
                                                                      [self.im_height, self.im_width])

        # input_image, ref_image = random_flip(input_image, ref_image)

        return cropped_ref_image, cropped_mid_image, cropped_input_image


class HEVC_ClassADataSet(data.Dataset):
    def __init__(self, root=testpath + "/HEVC_ClassA/images", filelist=testpath + "/HEVC_ClassA/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['PeopleOnStreet', 'Trafficive']
        AllIbpp = self.getbpp(refdir)

        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[3:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, orig, 'img' + str(i * 10 + j + 1).zfill(6) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'mbt2018L8':  # ['PeopleOnStreet','Trafficive']
            print('use mbt2018L8')
            Ibpp = ['1.0906005208333336', '1.1692015625']



        elif ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = ['0.8164710937499999', '1.0645114583333333']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['0.59288828125', '0.7457752604166666']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.4291872395833333', '0.5245141927083332']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.3078380208333333', '0.37245260416666665']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64

        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            input_image = np.asarray(input_image)

            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class HEVC_ClassBDataSet(data.Dataset):
    def __init__(self, root=testpath + "/HEVC_ClassB/images", filelist=testpath + "/HEVC_ClassB/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['BQTerrace', 'BasketballDrive', 'Cactus', 'Kimono1', 'ParkScene']
        AllIbpp = self.getbpp(refdir)

        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[2:-4]))
            cnt = 100
            # for im in imlist:
            #     if im[-4:] == '.png':
            #         cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, orig, 'im' + str(i * 10 + j + 1).zfill(5) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None

        if ref_i_folder == 'mbt2018L8':  # ['BQTerrace','BasketballDrive','Cactus','Kimono1','ParkScene']
            print('use mbt2018L8')
            Ibpp = ['2.0352541775173614', '1.2395263671875005', '1.6615885416666665', '0.8131320529513889',
                    '1.505930582682292']
        elif ref_i_folder == 'mbt2018L6':  # ['BQTerrace','BasketballDrive','Cactus','Kimono1','ParkScene']
            print('use mbt2018L6')
            Ibpp = ['0.8737985568576385', '0.40623828125000006', '0.6173167317708335', '0.31762288411458334',
                    '0.6914618598090279']
        elif ref_i_folder == 'mbt2018L4':  # ['BQTerrace','BasketballDrive','Cactus','Kimono1','ParkScene']
            print('use mbt2018L4')
            Ibpp = ['0.3567092556423611', '0.16953971354166664', '0.266955078125', '0.150726318359375',
                    '0.3117146809895834']
        elif ref_i_folder == 'mbt2018L2':  # ['BQTerrace','BasketballDrive','Cactus','Kimono1','ParkScene']
            print('use mbt2018L2')
            Ibpp = ['0.17544894748263892', '0.08332845052083333', '0.12454361979166671', '0.07128567165798612',
                    '0.1295003255208333']


        elif ref_i_folder == 'bmshj2018_hyperpriorL8':  # ['BQTerrace','BasketballDrive','Cactus','Kimono1','ParkScene']
            print('usefo bmshj2018_hyperpriorL8')
            Ibpp = ['2.151012641059028', '1.4562347005208336', '1.834372395833333', '0.9582505967881946',
                    '1.604661051432292']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'bmshj2018_hyperpriorL7':
            print('bmshj2018_hyperpriorL7')
            Ibpp = ['1.5100697157118053', '0.866876953125', '1.1564384765624998', '0.5498182508680557',
                    '1.0882934570312501']  # you need to fill bpps after generating crf=23


        elif ref_i_folder == 'H265L20':  # ['BQTerrace','BasketballDrive','Cactus','Kimono1','ParkScene']
            print('use H265L20')
            Ibpp = ['1.542864223125854', '1.1633543326184639', '1.371733642578125', '0.851354471842448',
                    '1.4042780558268229']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['1.060283990245048', '0.6547827627144608', '0.850405517578125', '0.5083706325954861',
                    '0.9498065524631075']
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.6295430167776639', '0.35587190116932194', '0.5058738606770833', '0.31582522922092016',
                    '0.6284449259440105']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.38665437958931015', '0.2179831112132353', '0.32359228515625', '0.21475524902343748',
                    '0.4215030246310764']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64

        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            input_image = np.asarray(input_image)

            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class HEVC_ClassCDataSet(data.Dataset):
    def __init__(self, root="C:/Users/sunhu/Documents/Data/test/HEVC/images", filelist="./data/C.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['BQMall', 'BasketballDrill', 'PartyScene', 'RaceHorses']
        AllIbpp = self.getbpp(refdir)
        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[2:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, orig, 'im' + str(i * 10 + j + 1).zfill(5) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'mbt2018L8':  # 'BQMall','BasketballDrill','PartyScene','RaceHorses'
            print('use mbt2018L8')
            Ibpp = ['1.4336753090659344', '1.5880391483516483', '2.6631473214285712', '1.6759214743589748']
        elif ref_i_folder == 'mbt2018L6':  # 'BQMall','BasketballDrill','PartyScene','RaceHorses'
            print('use mbt2018L6')
            Ibpp = ['0.7128720238095236', '0.752718063186813', '1.5579086538461535', '0.9155076694139194']
        elif ref_i_folder == 'mbt2018L4':  # 'BQMall','BasketballDrill','PartyScene','RaceHorses'
            print('use mbt2018L4')
            Ibpp = ['0.36315533424908414', '0.3471943681318681', '0.7823677884615385', '0.4549135760073259']
        elif ref_i_folder == 'mbt2018L2':  # 'BQMall','BasketballDrill','PartyScene','RaceHorses'
            print('use mbt2018L2')
            Ibpp = ['0.18137877747252754', '0.16027987637362634', '0.36145947802197803', '0.1973300137362637']



        elif ref_i_folder == 'H265L20':  # 'BQMall','BasketballDrill','PartyScene','RaceHorses'
            print('use H265L20')
            Ibpp = ['1.2295528818005768', '1.3279524550204698', '1.9773873491704375',
                    '1.253125715430403']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['0.850196614348766', '0.9344779378097392', '1.4772859263897868',
                    '0.9022071027930404']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.6065053030985409', '0.6539488391510451', '0.6539488391510451',
                    '0.648652129120879']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.44216941485768324', '0.45145215538138345', '0.7907774287599656',
                    '0.4567322000915751']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class HEVC_ClassDDataSet(data.Dataset):
    def __init__(self, root=testpath + "/HEVC_ClassD/images", filelist=testpath + "/HEVC_ClassD/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['BQSquare', 'BasketballPass', 'BlowingBubbles', 'RaceHorses']
        AllIbpp = self.getbpp(refdir)
        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[2:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, orig, 'im' + str(i * 10 + j + 1).zfill(5) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'mbt2018L8':  # 'BQSquare','BasketballPass','BlowingBubbles','RaceHorses'
            print('use mbt2018L8')
            Ibpp = ['2.6863425925925926', '1.507803819444444', '2.7416666666666663', '1.9523148148148146']
        elif ref_i_folder == 'mbt2018L6':  # 'BQSquare','BasketballPass','BlowingBubbles','RaceHorses'
            print('use mbt2018L6')
            Ibpp = ['1.5922526041666665', '0.811345486111111', '1.5893576388888888', '1.1058015046296295']
        elif ref_i_folder == 'mbt2018L4':  # 'BQSquare','BasketballPass','BlowingBubbles','RaceHorses'
            print('use mbt2018L4')
            Ibpp = ['0.8140625', '0.4050086805555555', '0.7652864583333333', '0.5569878472222222']
        elif ref_i_folder == 'mbt2018L2':  # 'BQSquare','BasketballPass','BlowingBubbles','RaceHorses'
            print('use mbt2018L2')
            Ibpp = ['0.3927517361111112', '0.1938194444444445', '0.34069444444444436', '0.24777199074074072']



        elif ref_i_folder == 'H265L20':  # 'BQSquare','BasketballPass','BlowingBubbles','RaceHorses'
            print('use H265L20')
            Ibpp = ['2.113060678506375', '1.3075171909041394', '2.3144807836328973',
                    '1.484280960648148']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['1.6189858691939891', '0.9746221405228759', '1.7566295615468408',
                    '1.112163628472222']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['1.2121684312386156', '0.7200563385076252', '0.7200563385076252',
                    '0.8150390624999999']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.8806548127276866', '0.523158786083878', '0.9413871017156864',
                    '0.5796694155092593']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class HEVC_ClassEDataSet(data.Dataset):
    def __init__(self, root=testpath + "/HEVC_ClassE/images", filelist=testpath + "/HEVC_ClassE/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['FourPeople', 'Johnny', 'KristenAndSara']
        AllIbpp = self.getbpp(refdir)
        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[2:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, orig, 'im' + str(i * 10 + j + 1).zfill(5) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'mbt2018L8':  # 'FourPeople','Johnny','KristenAndSara'
            print('use mbt2018L8')
            Ibpp = ['0.7926651278409091', '0.6070833333333333', '0.6147904829545453']
        elif ref_i_folder == 'mbt2018L6':  # 'FourPeople','Johnny','KristenAndSara'
            print('use mbt2018L6')
            Ibpp = ['0.3876408617424243', '0.2640358664772727', '0.2850964725378789']
        elif ref_i_folder == 'mbt2018L4':  # 'FourPeople','Johnny','KristenAndSara'
            print('use mbt2018L4')
            Ibpp = ['0.2124502840909091', '0.12969164299242425', '0.15160925662878785']
        elif ref_i_folder == 'mbt2018L2':  # 'FourPeople','Johnny','KristenAndSara'
            print('use mbt2018L2')
            Ibpp = ['0.11481948390151515', '0.05943241003787878', '0.0797147253787879']



        elif ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = ['0.8790632398200757', '0.7990715258049244',
                    '0.8012579900568181']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['0.6016153231534092', '0.5050100615530303',
                    '0.5050100615530303']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.4144445430871212', '0.3075056226325757',
                    '0.3357535807291666']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.2954070490056818', '0.1950412819602273',
                    '0.22554672703598488']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class HEVC_ClassFDataSet(data.Dataset):
    def __init__(self, root=testpath + "/HEVC_ClassF/images", filelist=testpath + "/HEVC_ClassF/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['BasketballDrillText', 'ChinaSpeed', 'SlideEditing', 'SlideShow']
        AllIbpp = self.getbpp(refdir)
        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[2:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 10
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 10 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(10):
                    inputpath.append(os.path.join(root, seq, orig, 'im' + str(i * 10 + j + 1).zfill(5) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = ['1.367587299342814', '1.0997165934244792', '1.8303870738636363',
                    '0.5366372514204546']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['0.9788165266106443', '0.8610923258463542', '1.5671206202651515',
                    '0.4229284446022728']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.6982091514490412', '0.6646761067708334', '1.3368664180871213',
                    '0.3315928622159091']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.49198802117000645', '0.5057501220703124', '1.129674775094697',
                    '0.2586487926136364']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class UVGDataSet(data.Dataset):
    def __init__(self, root=testpath + "/UVG/images", filelist=testpath + "/UVG/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide']
        AllIbpp = self.getbpp(refdir)
        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[2:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 12 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, seq, orig, 'im' + str(i * 12 + j + 1).zfill(3) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None

        if ref_i_folder == 'mbt2018L8':  # 'Beauty','Bosphorus','HoneyBee','Jockey','ReadySteadyGo','ShakeNDry','YachtRide'
            print('use mbt2018L8')
            Ibpp = ['1.1624361979166664', '0.5100621744791665', '0.7384817708333331', '0.4840719401041669',
                    '0.7213427734374999', '0.88466796875', '0.6537180989583332']
        elif ref_i_folder == 'mbt2018L6':  # 'Beauty','Bosphorus','HoneyBee','Jockey','ReadySteadyGo','ShakeNDry','YachtRide'
            print('use mbt2018L6')
            Ibpp = ['0.30627734375000004', '0.2389527994791667', '0.30717578125', '0.1598557942708333',
                    '0.3733652343749999', '0.4008118489583334', '0.33193684895833336']
        elif ref_i_folder == 'mbt2018L4':  # 'Beauty','Bosphorus','HoneyBee','Jockey','ReadySteadyGo','ShakeNDry','YachtRide'
            print('use mbt2018L4')
            Ibpp = ['0.06775195312499997', '0.11657389322916664', '0.1635224609375', '0.07745214843750002',
                    '0.20156380208333335', '0.20101953125000002', '0.1747526041666667']
        elif ref_i_folder == 'mbt2018L2':  # 'Beauty','Bosphorus','HoneyBee','Jockey','ReadySteadyGo','ShakeNDry','YachtRide'
            print('use mbt2018L2')
            Ibpp = ['0.02568359375', '0.05513346354166668', '0.09048242187500001', '0.04022428385416668',
                    '0.1057197265625', '0.09713281249999998', '0.0853505859375']





        elif ref_i_folder == 'H265L20':  # 'Beauty','Bosphorus','HoneyBee','Jockey','ReadySteadyGo','ShakeNDry','YachtRide'
            print('use H265L20')
            Ibpp = ['1.213396484375', '0.6849548339843748', '0.8600716145833333', '0.6581201985677083',
                    '0.6985362955729166', '0.7548777669270834',
                    '0.6584032389322916']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['0.6781825358072916', '0.46543627929687503', '0.5208554687500001', '0.35492000325520834',
                    '0.4952764485677082', '0.5092376302083333',
                    '0.46510823567708337']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.31613256835937503', '0.3216608072916667', '0.33581868489583333', '0.19616520182291666',
                    '0.36005086263020836', '0.35370100911458335',
                    '0.33273413085937503']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.13072729492187501', '0.22285115559895832', '0.2382386881510417', '0.12693098958333335',
                    '0.265823486328125', '0.25438053385416665',
                    '0.23811816406250003']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class MCL_JCVDataSet(data.Dataset):
    def __init__(self, root=testpath + "/MCL_JCV/images", filelist=testpath + "/MCL_JCV/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        AllIbpp = self.getbpp(refdir)
        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[2:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 12 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, seq, orig, 'im' + str(i * 12 + j + 1).zfill(5) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'mbt2018L8':
            print('use mbt2018L8')
            Ibpp = ['0.48980577256944446', '0.27785780164930557', '0.6441121419270833', '1.0840006510416667',
                    '1.1941845703124998', '2.3653401692708327',
                    '1.3468587239583332', '0.7693636067708335', '1.6761653645833334', '1.0381171332465275',
                    '0.36564127604166674', '0.9919949001736111',
                    '1.4023206922743052', '0.7318277994791668', '1.50592041015625', '0.3700425889756944',
                    '0.6992757161458334', '1.0534521484375',
                    '0.7175442165798612', '0.594248046875', '0.19066243489583332', '0.5716878255208334',
                    '0.514599609375', '0.42532226562500003',
                    '1.3929085286458334', '0.38371175130208335', '0.7161173502604165', '0.3949503580729166',
                    '0.17999837239583333', '0.5999131944444445']
        elif ref_i_folder == 'mbt2018L6':
            print('use mbt2018L6')
            Ibpp = ['0.12584906684027783', '0.13726264105902777', '0.29420166015624993', '0.6063557942708334',
                    '0.37763671874999993', '0.8968977864583334',
                    '0.40369791666666666', '0.1956559244791667', '0.8613736979166665', '0.4995347764756944',
                    '0.18541937934027777', '0.49834526909722227',
                    '0.8213107638888889', '0.3052530924479167', '0.6768717447916668', '0.11965196397569446',
                    '0.2707763671875', '0.5576546223958333',
                    '0.3535875108506945', '0.2834977213541667', '0.07936686197916666', '0.2691715494791667',
                    '0.2556184895833333', '0.20927897135416668',
                    '0.7745670572916667', '0.16188151041666668', '0.3729302300347222', '0.1742133246527778',
                    '0.06854166666666667', '0.20519476996527777']
        elif ref_i_folder == 'mbt2018L4':
            print('use mbt2018L4')
            Ibpp = ['0.05169813368055556', '0.0757595486111111', '0.14424370659722222', '0.3108574761284722',
                    '0.16028320312499997', '0.07857259114583334',
                    '0.12142415364583332', '0.08381184895833334', '0.44425618489583324', '0.2510240342881944',
                    '0.09496120876736112', '0.2328640407986111',
                    '0.4484578450520833', '0.1537990993923611', '0.2998779296875', '0.05847574869791667',
                    '0.13585123697916668', '0.2710856119791667',
                    '0.17270914713541666', '0.13654459635416666', '0.03916666666666667', '0.14343912760416666',
                    '0.13354980468750002', '0.11075683593749999',
                    '0.4008447265625', '0.08347439236111114', '0.19174397786458333', '0.08829345703125001',
                    '0.03189615885416667', '0.0846896701388889']
        elif ref_i_folder == 'mbt2018L2':
            print('use mbt2018L2')
            Ibpp = ['0.02512749565972222', '0.04198676215277778', '0.07012261284722222', '0.14143473307291668',
                    '0.0791064453125', '0.013170572916666668',
                    '0.050032552083333334', '0.038777669270833336', '0.22615071614583337', '0.12946912977430555',
                    '0.04796142578125001', '0.10300157335069444',
                    '0.2336846245659722', '0.07824028862847222', '0.12756347656250003', '0.03044704861111111',
                    '0.06784667968750001', '0.11872233072916664',
                    '0.07930908203125', '0.06486165364583334', '0.020734049479166666', '0.07505696614583332',
                    '0.0662255859375', '0.05817057291666668',
                    '0.1992626953125', '0.04362250434027778', '0.0935546875', '0.04448513454861111',
                    '0.015587565104166668', '0.041351996527777775']




        elif ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = ['0.8810246394230768', '0.41454013922275645', '0.808793444511218', '0.7300552759415064',
                    '1.23033447265625', '2.0078864820075757',
                    '1.190487023555871', '1.197613340435606', '1.212457460345644', '0.7156262520032051 ',
                    '0.46037503756009623', '0.9827971629607374',
                    '1.2731748923277246', '0.7285462990785257', '1.2628061147836538', '0.8076794746594551',
                    '0.7154471842447917', '1.2305682558001896',
                    '0.7370858999399038', '0.8121826171875', '0.62086669921875', '0.5163822428385416',
                    '0.6072672526041667', '0.670718994140625',
                    '0.9364139811197918', '0.5424184945913462', '0.6995442708333334', '0.7256566756810897',
                    '0.7177579752604167', '0.9080954527243592']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['0.4569689628405449', '0.26916942107371794', '0.507778069911859', '0.5309670472756409',
                    '0.7111413204308712', '1.2665778882575756',
                    '0.6862970525568183', '0.6723447857481061', '0.87607421875', '0.5051973783052884',
                    '0.31794558794070515', '0.7131976787860577',
                    '0.9796605819310897', '0.4665439703525641', '0.8486882136418269', '0.41845922225560894',
                    '0.4231168619791667', '0.927933016690341',
                    '0.5172325721153846', '0.5929569128787879', '0.34174804687499993', '0.35497599283854164',
                    '0.4093452962239583', '0.38843953450520835',
                    '0.7063065592447917', '0.31165865384615377', '0.4942157451923077', '0.3945703751001603',
                    '0.4166438802083333', '0.5070049579326923']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.2074134239783654', '0.1872661884014423', '0.33167098607772433', '0.3772974258814103',
                    '0.3764230439157197', '0.6917314009232954',
                    '0.3684881036931818', '0.30857377485795456', '0.6311749082623106', '0.3581496018629808',
                    '0.22765236879006406', '0.508901116786859',
                    '0.7087571364182692', '0.3085496168870192', '0.5580178285256411', '0.21135629507211545',
                    '0.26908691406249996', '0.6860085227272728',
                    '0.3638534154647436', '0.4262639825994318', '0.18730143229166668', '0.2519954427083333',
                    '0.288729248046875', '0.23893107096354166',
                    '0.5201534016927084', '0.1943409455128205', '0.3497906024639423', '0.21839850010016024',
                    '0.21839850010016024', '0.29448711688701923']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.10561492137419871', '0.13578694661458332', '0.2280545748197116', '0.2624342698317308',
                    '0.22290002071496212', '0.2986557469223485',
                    '0.19638634883996212', '0.15446185487689393', '0.4566458037405303', '0.2551247621193911',
                    '0.16380490034054487', '0.3557673527644231',
                    '0.5119378505608974', '0.21402055789262822', '0.37146058693910255', '0.12751057942708333',
                    '0.18840494791666668', '0.49873342803030307',
                    '0.25531444060496794', '0.30181921756628793', '0.11124837239583332', '0.18346557617187503',
                    '0.208798828125', '0.1654435221354167',
                    '0.379847412109375', '0.13294051732772436', '0.24887820512820516', '0.13997051532451923',
                    '0.11552164713541666', '0.17997389573317304']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class VTLDataSet(data.Dataset):
    def __init__(self, root=testpath + "/VTL/images", filelist=testpath + "/VTL/originalv.txt",
                 refdir='L12000', testfull=False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = ['akiyo_cif', 'BigBuckBunny_CIF_24fps', 'bridge-close_cif', 'bridge-far_cif',
                          'bus_cif', 'coastguard_cif', 'container_cif', 'ElephantsDream_CIF_24fps',
                          'flower_cif', 'foreman_cif', 'hall_cif', 'highway_cif',
                          'mobile_cif', 'mother-daughter_cif', 'news_cif', 'paris_cif',
                          'silent_cif', 'stefan_cif', 'tempete_cif', 'waterfall_cif']
        AllIbpp = self.getbpp(refdir)
        ii = 0
        orig = 'original'
        relist = 'L20'
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = float(AllIbpp[ii])
            imlist = os.listdir(os.path.join(root, seq, orig))
            imlist.sort(key=lambda x: int(x[3:-4]))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, relist, refdir, 'im' + str(i * 12 + 1).zfill(4) + '.png')
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, seq, orig, 'img' + str(i * 12 + j + 1).zfill(6) + '.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = ['1.218140625', '1.9132539062499998', '2.0703242187499997', '1.6115859375',
                    '1.7069711538461536', '1.6291289062499998', '1.569703125', '1.215890625',
                    '2.122539992559523', '1.3840664062499999', '1.18468359375', '1.3180781249999998',
                    '2.61266015625', '0.9783671874999998', '1.2682773437500001', '1.90941796875',
                    '1.91764453125', '1.81357421875', '1.8894486860795459',
                    '2.3185458096590907']  # you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = ['0.8628789062499999', '1.4966289062500002', '1.51848046875', '1.03679296875',
                    '1.3293945312499997', '1.25010546875', '1.19199609375', '0.8740000000000001',
                    '1.7050037202380952', '1.01421484375', '0.8432539062500001', '0.8320156249999999',
                    '2.0933671874999997', '0.6965195312500001', '0.9431015624999999', '1.5132539062500001',
                    '1.4343242187499998', '1.4394897460937501', '1.459441583806818',
                    '1.788902698863636']  # you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = ['0.618890625', '1.134921875', '1.06659375', '0.61803515625',
                    '1.0085486778846153', '0.93127734375', '0.8756015625000001', '0.62198046875',
                    '1.3634114583333332', '0.7245742187499999', '0.5959453125', '0.5299765625',
                    '1.6597382812500001', '0.49790234375000003', '0.7001835937499998', '1.1840664062500001',
                    '1.02003515625', '1.13701171875', '1.1150124289772727',
                    '1.3283647017045457']  # you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = ['0.45659765624999993', '0.8331210937499999', '0.7309726562500001', '0.373109375',
                    '0.7420673076923078', '0.66645703125', '0.62825390625', '0.43990624999999994',
                    '1.0460007440476191', '0.51011328125', '0.4258203125', '0.3426171874999999',
                    '1.2754453125', '0.35722265625', '0.5267265624999999', '0.9151835937499999',
                    '0.6996484375', '0.87962646484375', '0.8198020241477274',
                    '0.9659934303977273']  # you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]),
                                    torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim

class HEVCDataSet(data.Dataset):  # 定义一个名为 HEVCDataSet 的类，它继承自 data.Dataset 类。
    def __init__(self, root="C:/Users/lenovo/Documents/Workspace/Data/test/HEVC_B", filelist="../data/B.txt", testfull=True):  # 定义类的构造函数，接受三个参数：root，filelist 和 testfull。
        with open(filelist) as f:  # 使用 with 语句打开 filelist 文件。
            folders = f.readlines()  # 读取文件中的所有行，并将其存储在 folders 列表中。
        self.input = []  # 初始化 input 列表。
        self.hevcclass = []  # 初始化 hevcclass 列表。
        for folder in folders:  # 遍历 folders 列表中的每个元素。
            seq = folder.rstrip()  # 删除字符串末尾的空白字符。
            imlist = os.listdir(os.path.join(root, seq))  # 获取指定目录下的文件列表。
            cnt = 0  # 初始化计数器 cnt。
            for im in imlist:  # 遍历文件列表中的每个文件。
                if im[-4:] == '.png':  # 如果文件的后缀为 .png，则计数器加一。
                    cnt += 1
                if cnt == 100:  # 如果计数器达到了 100，则跳出循环。
                    break
            if testfull:  # 如果 testfull 参数为真，则将帧范围设置为图像数量除以 10。
                framerange = cnt // 10
            else:  # 否则将帧范围设置为 1。
                framerange = 1
            for i in range(framerange):  # 遍历帧范围内的每一帧。
                inputpath = []  # 初始化 inputpath 列表。
                for j in range(10):  # 遍历已此帧为起始帧前十张图像。
                    inputpath.append(os.path.join(root, seq, 'im' + str(i * 10 + j + 1).zfill(5) + '.png'))  # 将图像的路径添加到 inputpath 列表中。
                self.input.append(inputpath)  # 将 inputpath 列表添加到 input 列表中。

    def __len__(self):  # 定义 __len__ 方法，返回输入列表的长度。
        return len(self.input) # self.input (40, 10)

    def __getitem__(self, index):  # 定义 __getitem__ 方法，接受一个索引参数，并返回该索引处的输入图像。
        input_images = []  # 初始化 input_images 列表。
        for filename in self.input[index]:  # 遍历输入列表中指定索引处的元素（即输入路径列表）中的每个元素（即输入路径）。
            # print(filename)
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0  # 使用 imageio 库读取图像文件，并将其转换为浮点类型并归一化到 [0,1] 范围内。然后使用 transpose 方法将图像矩阵转置，使其形状变为 [3, h, w]（即通道数在前，高度和宽度在后）。
            h = int((input_image.shape[1] // 64) * 64)   # 计算图像高度 h，使其能够被64整除
            w = int((input_image.shape[2] // 64) * 64)   # 计算图像宽度 w，使其能够被64整除
            input_images.append(input_image[:, :h, :w])  # 将图像裁剪到指定大小，并添加到 input_images 列表中。

        input_images = np.array(input_images)  # 将 input_images 列表转换为 numpy 数组并返回。

        return input_images # shape(10, 3, h, w)
