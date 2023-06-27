import torch

from fvc_net.basic import *
from fvc_net.models.google import CompressionModel, get_scale_table
from fvc_net.decoder_resblock import *
from fvc_net.encoder_resblock import *
from fvc_net.hyper_decoder import *
from fvc_net.hyper_encoder import *

from fvc_net.hyper_prior import *
from fvc_net.layers.layers import MaskedConv2d, subpel_conv3x3
from fvc_net.entropy_models import GaussianConditional,GaussianConditional_HAMC

from fvc_net.models.utils import (
    conv,
    deconv,
    quantize_ste,
    update_registered_buffers,
)

def get_downsampled_shape(height, width, p):

    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)

class Hyperprior_flow(CompressionModel):
    def __init__(self, planes: int = 192, mid_planes: int = 192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        out_channel_mv = 128
        self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
        self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
        self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
        self.gaussian_conditional = GaussianConditional_HAMC(None)


        self.auto_regressive_mv = MaskedConv2d(out_channel_mv, out_channel_mv, kernel_size=5, padding=2, stride=1)
        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(out_channel_mv * 2 , out_channel_mv , 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv , out_channel_mv, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv , out_channel_mv* 2 , 1),
        )

    def forward(self, y):
        z = self.hyper_encoder(y)                           # y (4,128,16,16)  z(4,128,4,4)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)   # z_hat (4,128,4,4)

        scales = self.hyper_decoder_scale(z_hat)            # scales(4,128,16,16)
        means  = self.hyper_decoder_mean(z_hat)             # means(4,128,16,16)

        y_hat = quantize_ste(y - means) + means             # y_hat (4,128,16,16)

        ###auto-regressive spatial correlation
        ctx_params_mv = self.auto_regressive_mv(y_hat)

        gaussian_params_mv = self.entropy_parameters_mv(torch.cat((scales, ctx_params_mv), dim=1))

        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)

        # scales = self.hyper_decoder_ctx_params_mv_scale(torch.cat((scales, ctx_params_mv), dim=1))
        # means = self.hyper_decoder_mean(torch.cat((scales, ctx_params_mv), dim=1))

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat_mv, means_hat_mv)


        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y):
        z = self.hyper_encoder(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)

        y_string = self.gaussian_conditional.compress(y, indexes, means)
        # y_string = self.gaussian_conditional.compress(y, indexes)


        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

        # y_hat = self.gaussian_conditional.quantize(y, "dequantize")
        # y_hat = self.gaussian_conditional.decompress(y_string, indexes, z_hat.dtype, means)

        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)

        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        return y_hat


class Hyperprior_mv(CompressionModel):
    def __init__(self, planes: int = 192, mid_planes: int = 192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        out_channel_mv = 128
        self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
        self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
        self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
        self.gaussian_conditional = GaussianConditional_HAMC(None)



    def forward(self, y):
        z = self.hyper_encoder(y)                           # y (4,128,16,16)  z(4,128,4,4)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)   # z_hat (4,128,4,4)

        scales = self.hyper_decoder_scale(z_hat)            # scales(4,128,16,16)
        means = self.hyper_decoder_mean(z_hat)             # means(4,128,16,16)





        _, y_likelihoods = self.gaussian_conditional(y, scales, means)

        y_hat = quantize_ste(y - means) + means  # y_hat (4,128,16,16)


        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y):
        z = self.hyper_encoder(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)

        y_string = self.gaussian_conditional.compress(y, indexes, means)
        # y_string = self.gaussian_conditional.compress(y, indexes)

        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

        # y_hat = self.gaussian_conditional.quantize(y, "dequantize")
        # y_hat = self.gaussian_conditional.decompress(y_string, indexes, z_hat.dtype, means)

        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)

        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        return y_hat


class Hyperprior(CompressionModel):
    def __init__(self, planes: int = 192, mid_planes: int = 192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        out_channel_mv = 128
        self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
        self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
        self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
        self.gaussian_conditional = GaussianConditional(None)

        # self.auto_regressive = MaskedConv2d(out_channel_M, out_channel_M, kernel_size=5, padding=2, stride=1)
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 2, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M*2, 1),
        )
    def forward(self, y,temporal_prior_params):
        z = self.hyper_encoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales = self.hyper_decoder_scale(z_hat)

        gaussian_params = self.entropy_parameters(torch.cat((temporal_prior_params, scales), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)

        y_hat = quantize_ste(y - means_hat) + means_hat
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y,temporal_prior_params):
        z = self.hyper_encoder(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)


        gaussian_params = self.entropy_parameters(torch.cat((temporal_prior_params, scales), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_string = self.gaussian_conditional.compress(y, indexes, means_hat)
        # y_string = self.gaussian_conditional.compress(y, indexes)

        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means_hat)

        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}


    def decompress(self, strings, shape,temporal_prior_params):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        gaussian_params = self.entropy_parameters(torch.cat((temporal_prior_params, scales), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means_hat)

        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        return y_hat