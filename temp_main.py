import torch
from fvc import FVC_base
from test_net.test_net import testHEVC, testhevcC

if __name__ == "__main__":
    # args = parse_args()
    # I_codec = cheng2020_anchor(quality=lambda_I_quality_map[args.lambda_weight], metric='mse', pretrained=True).cuda()
    # I_codec.eval()
    model_path = 'model/2048/old-i-stage/iter242295.model'
    model = FVC_base()
    pretrained_dict = torch.load(model_path)

    # 加载模型并滤除无用参数
    model_dict = model.state_dict()
    ckpt = pretrained_dict
    pretrained_net = {k: v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(pretrained_net)
    model.load_state_dict(model_dict)

    net = model.cuda()

    # testhevcC(global_step=0, net=net, ref_i_dir="H265L20", testfull=True)
    testHEVC(global_step=0, net=net, filelist="./data/B.txt", testfull=True)

    # print("Number of Total Parameters:", sum(x.numel() for x in net.parameters()))
    # global test_dataset
    # test_dataset = HEVCDataSet()
    # test()
    # exit(0)
