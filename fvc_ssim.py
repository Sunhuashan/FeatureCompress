from typing import List

import torch
import torch.nn as nn
from fvc_net.models.utils import (conv,deconv,quantize_ste,update_registered_buffers,)

from fvc_net.feature import *
from fvc_net.motion import *
from fvc_net.decoder_resblock import *
from fvc_net.encoder_resblock import *
from fvc_net.hyper_decoder import *
from fvc_net.hyper_encoder import *
from fvc_net.hyper_prior import *
from fvc_net.layers.layers import GDN, MaskedConv2d

from fvc_net.ms_ssim_torch import *
from mmcv.ops import ModulatedDeformConv2d as DCN
from video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1

import dataset

def save_model(model, iter):
    torch.save(model.state_dict(), "./model/iter{}.model".format(iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0




class FVC_ssim(nn.Module):
    def __init__(self,):
        super().__init__()
        self.out_channel_mv=128
        self.out_channel_F = out_channel_F
        self.out_channel_O = out_channel_O
        self.out_channel_M = out_channel_M
        self.deform_ks = deform_ks
        self.deform_groups = deform_groups


        self.feature_extract = feature_exactnet()
        self.frame_reconstruct = feature_reconsnet()

        self.motion_estimate = motion_estimation()
        self.motion_compensate = motion_compensate()

        ###warp
        self.offset_mask_conv = nn.Conv2d(out_channel_O,deform_groups * 3 * deform_ks * deform_ks,kernel_size=deform_ks,stride=1,padding=1,)
        self.deform_conv = DCN(out_channel_F, out_channel_F, kernel_size=(deform_ks, deform_ks), padding=deform_ks // 2,deform_groups=deform_groups)

        ###motion compression
        self.motion_encoder = EncoderWithResblock(in_planes=out_channel_O, out_planes=out_channel_M)
        self.motion_decoder = DecoderWithResblock(in_planes=out_channel_M, out_planes=out_channel_O)
        self.motion_hyperprior = Hyperprior_mv(planes=out_channel_M, mid_planes=out_channel_M)

        ###context compression
        self.res_encoder = EncoderWithResblock(in_planes=out_channel_M, out_planes=128)
        self.res_decoder = DecoderWithResblock(in_planes=128, out_planes=out_channel_F)
        self.res_hyperprior = Hyperprior(planes=128, mid_planes=128)

        ###context
        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_O, out_channel_O, 5, stride=2, padding=2),
            GDN(out_channel_O),
            nn.Conv2d(out_channel_O, out_channel_O, 5, stride=2, padding=2),
            GDN(out_channel_O),
            nn.Conv2d(out_channel_O, out_channel_O, 5, stride=2, padding=2),
            GDN(out_channel_O),
            nn.Conv2d(out_channel_O, out_channel_O, 5, padding=2),
            nn.Conv2d(out_channel_O, out_channel_M, 5, padding=2),
        )


        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N*2, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),

            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),

            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            # nn.Conv2d(out_channel_N, out_channel_M, 5,stride=2, padding=2),
            nn.Conv2d(out_channel_N, out_channel_M, 5, padding=2),
        )

        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),

            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),

            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            # subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N * 2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),

        )



    def dcn_warp(self, offset_info, f_ref):
        offset_and_mask = self.offset_mask_conv(offset_info)
        o1, o2, mask = torch.chunk(offset_and_mask, 3, dim=1)
        offset_map = torch.cat((o1, o2), dim=1)

        mask = torch.sigmoid(mask)
        return self.deform_conv(f_ref, offset_map, mask)

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list


    def forward(self,input_image, referframe):

        #feature extract
        f_cur = self.feature_extract(input_image)
        f_ref = self.feature_extract(referframe)
        f = torch.cat((f_cur, f_ref), dim=1)

        # motion estimation
        offset = self.motion_estimate(f)

        ###motion compression
        # encode motion info
        offset = self.motion_encoder(offset)
        offset_hat, motion_likelihoods = self.motion_hyperprior(offset)
        # decode motion info
        offset_info = self.motion_decoder(offset_hat)   #offset_info (4,64,128,128)

        # motion compensation
        deformed_f_ref = self.dcn_warp(offset_info, f_ref)

        f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        f_mc = self.motion_compensate(f_mc)
        f_context = deformed_f_ref + f_mc                                         ###prediction (4,64,128,128)

        f_temporal_prior_params = self.temporalPriorEncoder(f_context)

        ### feature space context
        # feature = self.contextualEncoder(torch.cat((f_cur, f_context), dim=1))
        encoded_res = self.res_encoder(torch.cat((f_cur, f_context), dim=1))
        # encoded_res = self.res_encoder(feature)
        f_res_hat, f_res_likelihoods = self.res_hyperprior(encoded_res,f_temporal_prior_params)

        f_recon_image = self.contextualDecoder_part1(f_res_hat)


        recon_image = self.contextualDecoder_part2(torch.cat((f_recon_image, f_context), dim=1))

        batch_size = encoded_res.size()[0]

        # # feature space residual
        # res = f_cur - f_context
        #
        # encoded_res = self.res_encoder(res)
        # f_res_hat, f_res_likelihoods = self.res_hyperprior(encoded_res)
        #
        # f_res = self.res_decoder(f_res_hat)
        #
        # batch_size = encoded_res.size()[0]
        #
        # # frame construction
        # f_combine = f_res + f_context
        #
        # x_rec = self.frame_reconstruct(f_combine)

        x_rec = self.frame_reconstruct(recon_image)

        clipped_recon_image = x_rec.clamp(0., 1.)


        ####Rate_LOSS
        ms_ssim_loss = ms_ssim(input_image.cuda(), clipped_recon_image.cuda(), data_range=1.0,size_average=True)

        # ms_ssim_loss = MS_SSIM(input_image.cuda(), clipped_recon_image.cuda(), data_range=1.0, size_average=True)

        # mse_loss = torch.mean((x_rec - input_image).pow(2))
        im_shape = input_image.size()


        bpp_mv = torch.log(motion_likelihoods['y']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_mvprior = torch.log(motion_likelihoods['z']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])

        bpp_res = torch.log(f_res_likelihoods['y']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_resprior = torch.log(f_res_likelihoods['z']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])

        bpp = bpp_mv + bpp_mvprior + bpp_res + bpp_resprior


        return clipped_recon_image, ms_ssim_loss,  bpp_res, bpp_resprior, bpp_mv, bpp_mvprior, bpp


    def compress(self,input_image,refer_frame):
        f_cur = self.feature_extract(input_image)
        f_ref = self.feature_extract(refer_frame)
        f = torch.cat((f_cur, f_ref), dim=1)
        # motion estimation
        offset = self.motion_estimate(f)
        # encode motion info
        offset = self.motion_encoder(offset)


        offset_hat, out_motion = self.motion_hyperprior.compress(offset)
        # decode motion info
        offset_info = self.motion_decoder(offset_hat)
        # motion compensation
        deformed_f_ref = self.dcn_warp(offset_info, f_ref)
        f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        f_mc = self.motion_compensate(f_mc)
        f_context = deformed_f_ref + f_mc  ###prediction (4,64,128,128)

        f_temporal_prior_params = self.temporalPriorEncoder(f_context)

        ### feature space context
        encoded_res = self.res_encoder(torch.cat((f_cur, f_context), dim=1))

        # encoded_res = self.res_encoder(feature)
        f_res_hat, out_context = self.res_hyperprior.compress(encoded_res, f_temporal_prior_params)


        f_recon_image = self.contextualDecoder_part1(f_res_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((f_recon_image, f_context), dim=1))

        batch_size = encoded_res.size()[0]

        # frame construction
        x_rec = self.frame_reconstruct(recon_image)

        clipped_recon_image = x_rec.clamp(0., 1.)

        return clipped_recon_image,{
            "strings": {
                "motion": out_motion["strings"],
                "context": out_context["strings"],
            },
            "shape": {
                "motion": out_motion["shape"],
                "context": out_context["shape"]},
        }

    def decompress(self,ref_image,strings,shapes):
        # motion


        key = "motion"
        offset_hat = self.motion_hyperprior.decompress(strings[key], shapes[key])
        offset_info = self.motion_decoder(offset_hat)

        f_ref = self.feature_extract(ref_image)
        deformed_f_ref = self.dcn_warp(offset_info, f_ref)
        f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        f_mc = self.motion_compensate(f_mc)
        f_context = deformed_f_ref + f_mc  ###prediction (4,64,128,128)

        f_temporal_prior_params = self.temporalPriorEncoder(f_context)

        # context
        key = "context"

        f_res_hat = self.res_hyperprior.decompress(strings[key], shapes[key],f_temporal_prior_params)

        f_recon_image = self.contextualDecoder_part1(f_res_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((f_recon_image, f_context), dim=1))

        x_rec = self.frame_reconstruct(recon_image)
        x_rec= x_rec.clamp(0., 1.)
        return {"x_hat": x_rec}


    def load_state_dict(self, state_dict):

        # Dynamically update the entropy bottleneck buffers related to the CDFs

        update_registered_buffers(
            self.res_hyperprior.gaussian_conditional,
            "res_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.res_hyperprior.entropy_bottleneck,
            "res_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        update_registered_buffers(
            self.motion_hyperprior.gaussian_conditional,
            "motion_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.motion_hyperprior.entropy_bottleneck,
            "motion_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):

        SCALES_MIN = 0.11
        SCALES_MAX = 256
        SCALES_LEVELS = 64

        def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
            return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

        if scale_table is None:
            scale_table = get_scale_table()
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.res_hyperprior.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.res_hyperprior.entropy_bottleneck.update(force=force)

        updated |= self.motion_hyperprior.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.motion_hyperprior.entropy_bottleneck.update(force=force)

        return updated


    def test(self,input_image, ref_image):

        x_rec,strings_and_shape = self.compress(input_image, ref_image)

        strings, shape = strings_and_shape["strings"], strings_and_shape["shape"]

        reconframe = self.decompress(ref_image, strings, shape)["x_hat"]

        num_pixels = input_image.size()[2] * input_image.size()[3]
        num_pixels = torch.tensor(num_pixels).float()
        mv_y_string=strings["motion"][0][0]
        mv_z_string=strings["motion"][1][0]
        res_y_string=strings["context"][0][0]
        res_z_string=strings["context"][1][0]
        bpp = (len(mv_y_string) + len(mv_z_string) + len(res_y_string) + len(res_z_string)) * 8.0 / num_pixels

        reconframe = reconframe.clamp(0., 1.)
        # cpr_mse_loss=torch.mean((x_rec - input_image).pow(2))
        # mse_loss = torch.mean((reconframe - input_image).pow(2))
        # decompressreconframepsnr = torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cuda().detach()
        # compressreconframepsnr=torch.mean(10 * (torch.log(1. / cpr_mse_loss) / np.log(10))).cuda().detach()
        # print('compress_reconframe:', compressreconframepsnr)
        # print('decompress_reconframe:',decompressreconframepsnr)

        return bpp,reconframe











