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
from fvc_net.convlstm import ConvLSTM
from fvc_net.hyper_prior import *
from fvc_net.layers.layers import GDN, MaskedConv2d
from fvc_net.layer import *

from mmcv.ops import ModulatedDeformConv2d as DCN
from video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1

import dataset

def save_model(model,iter, train_lambda,stages):
    torch.save(model.state_dict(), "C:/Users/lenovo/Documents/Workspace/FeatureCompress/model/{}/{}-stage/iter-{}.model".format(train_lambda,stages,iter))

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




class net(nn.Module):
    def __init__(self,):
        super().__init__()
        self.out_channel_mv= 128
        self.out_channel_F = out_channel_F
        self.out_channel_O = out_channel_O
        self.out_channel_M = out_channel_M
        self.deform_ks = deform_ks
        self.deform_groups = deform_groups


        self.n_c = 128
        self.lstm_layer_num = 4
        self.hidden_dim = [self.n_c // 2 for i in range(self.lstm_layer_num)]
        self.conv_lstm = ConvLSTM(self.n_c // 2, self.hidden_dim, (5, 5), self.lstm_layer_num, True, True, True)

        self.feature_extraction = FeatureExtractor()
        self.frame_reconstruct=feature_reconsnet()
        self.ME_Net = motion_estimation()
        self.MC_Net=motion_compensate()

        self.adap = DAB(n_feat=64, kernel_size=3, reduction=2, aux_channel=64)


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

    def forward(self, input_image, ref_list):
        # ref_list: (b, t, c, h, w)
        referframe = ref_list[:, -1]
        seq_len = ref_list.size(1)

        f_cur1, f_cur2, f_cur3 = self.feature_extraction(input_image)
        f_ref1, f_ref2, f_ref3 = self.feature_extraction(referframe)

        #feature extract
        offset1 = self.ME_Net(torch.cat([f_cur1, f_ref1], dim=1))
        offset2 = self.ME_Net(torch.cat([f_cur2, f_ref2], dim=1))
        offset3 = self.ME_Net(torch.cat([f_cur3, f_ref3], dim=1))

        offset = self.adap(offset1, offset2, offset3)

        # hyper motion estimation
        # hyper_motion = self.ME_Net(torch.cat([offset, refermotion], dim=1))
        # encode_hyper_motion = self.motion_encoder(hyper_motion)
        # hyper_motion_hat, hyper_motion_likelihoods = self.motion_hyperprior(encode_hyper_motion)
        # decode_hyper_motion = self.motion_decoder(hyper_motion_hat)

        # hyper motion compensation
        # deformed_motion = self.dcn_warp(decode_hyper_motion, refermotion)
        # predict_motion = self.MC_Net(torch.cat([deformed_motion, refermotion], dim=1))
        #
        # # motion residual
        # motion_res = offset - predict_motion
        # encode_motion_res = self.motion_encoder(motion_res)
        # motion_res_hat, motion_res_likelihoods = self.motion_hyperprior(encode_motion_res)
        # decode_motion_res = self.motion_decoder(motion_res_hat)
        #
        # offset_info = predict_motion + decode_motion_res

        # motion compression
        # encode motion info
        offset = self.motion_encoder(offset)
        offset_hat, motion_likelihoods = self.motion_hyperprior(offset)
        # decode motion info
        offset_info = self.motion_decoder(offset_hat)   #offset_info (4,64,128,128)

        # 此处是否可以添加 mv_refine 网络

        # motion compensation
        deformed_f_ref = self.dcn_warp(offset_info, f_ref1)

        f_mc = torch.cat((deformed_f_ref, f_ref1), dim=1)
        f_mc = self.MC_Net(f_mc)

        # hidden_context
        feature_pool = []
        h_pool = []

        for t in range(seq_len):
            feature_t1, feature_t2, feature_t3 = self.feature_extraction(ref_list[:, t])
            h_pool.append(feature_t1)
            feature_pool.append([feature_t1, feature_t2, feature_t3 ])
        h_pool = torch.stack(h_pool, dim=1)

        layer_output, last_state = self.conv_lstm(h_pool)
        hidden_feature = layer_output[-1][:, t]

        f_context = deformed_f_ref + f_mc      # prediction (4,64,128,128)

        f_context = self.adap(f_context, hidden_feature)

        f_temporal_prior_params = self.temporalPriorEncoder(f_context)

        # feature space context
        # feature = self.contextualEncoder(torch.cat((f_cur, f_context), dim=1))
        encoded_res = self.res_encoder(torch.cat((f_cur1, f_context), dim=1))
        # encoded_res = self.res_encoder(feature)
        f_res_hat, f_res_likelihoods = self.res_hyperprior(encoded_res, f_temporal_prior_params)

        f_recon_image = self.contextualDecoder_part1(f_res_hat)

        recon_image = self.contextualDecoder_part2(torch.cat((f_recon_image, f_context), dim=1))

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

        mse_loss = torch.mean((x_rec - input_image).pow(2))
        pred_loss = torch.mean((f_cur1 - f_context).pow(2))
        batch_size, c, h, w = input_image.size()

        bpp_mv = torch.log(motion_likelihoods['y']).sum() / (-math.log(2) * batch_size * h * w)
        bpp_mv_z = torch.log(motion_likelihoods['z']).sum() / (-math.log(2) * batch_size * h * w)

        bpp_y = torch.log(f_res_likelihoods['y']).sum() / (-math.log(2) * batch_size * h * w)
        bpp_z = torch.log(f_res_likelihoods['z']).sum() / (-math.log(2) * batch_size *h * w)

        bpp = bpp_mv + bpp_mv_z + bpp_y + bpp_z

        return clipped_recon_image, mse_loss, bpp_y, bpp_z, bpp_mv, bpp_mv_z, bpp, pred_loss

    def compress(self,input_image,refer_frame):
        f_cur1, f_cur2, f_cur3 = self.feature_extraction(input_image)

        f_ref1, f_ref2, f_ref3 = self.feature_extraction(refer_frame)

        # feature extract
        offset1 = self.ME_Net(torch.cat([f_cur1, f_ref1], dim=1))
        offset2 = self.ME_Net(torch.cat([f_cur2, f_ref2], dim=1))
        offset3 = self.ME_Net(torch.cat([f_cur3, f_ref3], dim=1))

        offset = self.adap(offset1, offset2, offset3)


        # encode motion info
        offset = self.motion_encoder(offset)


        offset_hat, out_motion = self.motion_hyperprior.compress(offset)
        # decode motion info
        offset_info = self.motion_decoder(offset_hat)
        # motion compensation
        deformed_f_ref = self.dcn_warp(offset_info, f_ref1)
        f_mc = torch.cat((deformed_f_ref, f_ref1), dim=1)
        f_mc = self.MC_Net(f_mc)
        f_context = deformed_f_ref + f_mc  ###prediction (4,64,128,128)

        f_temporal_prior_params = self.temporalPriorEncoder(f_context)

        ### feature space context
        encoded_res = self.res_encoder(torch.cat((f_cur1, f_context), dim=1))

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

        f_ref1, _, _ = self.feature_extraction(ref_image)

        deformed_f_ref = self.dcn_warp(offset_info, f_ref1)
        f_mc = torch.cat((deformed_f_ref, f_ref1), dim=1)
        f_mc = self.MC_Net(f_mc)
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











