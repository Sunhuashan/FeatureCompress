from fvc_ssim import *

torch.backends.cudnn.enabled = True

from torchvision import transforms
from fvc_net.datasets import VideoFolder
from test_net.test_net import *
from fvc_net.zoo.image import cheng2020_anchor,mbt2018,bmshj2018_hyperprior



# gpu_num = 4
gpu_num = torch.cuda.device_count()       #  * gpu_num
cur_lr = base_lr = 5e-6
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0#  // gpu_num
gpu_per_batch = 2
test_step = 10000#  // gpu_num
tot_epoch = 1000000
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)

clip_max_norm=1.0
aux_learning_rate=1e-4

parser = argparse.ArgumentParser(description='FVC reimplement')

parser.add_argument('-l', '--log', default='',help='output training details')
parser.add_argument('-p', '--pretrain', default = '',help='load pretrain model')

parser.add_argument('--test', action='store_true')
parser.add_argument('--testuvg', action='store_true')
parser.add_argument('--testvtl', action='store_true')
parser.add_argument('--testmcl', action='store_true')
parser.add_argument('--testauc', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', required=True,help = 'hyperparameter of Reid in json format')

# traindata_path=r'C:\Users\Administrator\Desktop\vimeo_septuplet\test.txt'
# traindata_path='F:/vc_project/video_compression/data/train_data/vimeo_septuplet/test.txt'
traindata_path='G:/wangyiming/data/train_data/vimeo_septuplet/test.txt'



# testdata_path='F:/vc_project/video_compression/data/train_data/vimeo_septuplet'
testdata_path='G:/wangyiming/data/train_data/vimeo_septuplet/'


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

def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    else:
        lr = base_lr * (lr_decay ** (global_step // decay_interval))
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Var(x):
    return Variable(x.cuda())


def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()
    return aux_loss_sum


def configure_optimizers(net, base_lr,aux_lr):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)),lr=base_lr,)

    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)),lr=aux_lr,)
    return optimizer, aux_optimizer



def train(epoch, global_step):
    global gpu_per_batch
    global optimizer, aux_optimizer
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    summsssim=0
    sumbpp = 0
    sumbpp_feature = 0
    sumbpp_z = 0
    sumbpp_mv = 0
    sumbpp_mv_z=0

    # onestage_train_loader = DataLoader(dataset = onestage_train_dataset, batch_size=gpu_per_batch, num_workers=gpu_num, shuffle=True,pin_memory=True, )
    twostage_train_loader=DataLoader(dataset = twostage_train_dataset, batch_size=gpu_per_batch, num_workers=gpu_num, shuffle=True,pin_memory=True, )
    net.train()
    tot_iter = len(twostage_train_loader)


    for batch_idx, input in enumerate(twostage_train_loader):

        global_step += 1
        bat_cnt += 1

        distortionloss=[]
        rateloss=[]
        bpp_feature_loss=[]
        bpp_z_loss=[]
        bpp_mv_loss=[]
        bpp_mv_z_loss=[]

        d = [frames.cuda() for frames in input]
        with torch.no_grad():
            frame_i_out = net_i(d[0])
            d[0] = frame_i_out["x_hat"]
            ref_image=d[0]

        for i in range(1,len(d)):
            # global_step += 1
            # bat_cnt += 1

            input_image= Var(d[i])
            ref_image, ms_ssim,  bpp_feature, bpp_z, bpp_mv,bpp_mv_z, bpp =net(input_image, ref_image)
            ref_image=ref_image.detach()

            ms_ssim,  bpp_feature, bpp_z, bpp_mv,bpp_mv_z, bpp = torch.mean(ms_ssim),  torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv),torch.mean(bpp_mv_z), torch.mean(bpp)

            distortionloss.append(ms_ssim)
            rateloss.append(bpp)
            bpp_feature_loss.append(bpp_feature)
            bpp_z_loss.append(bpp_z)
            bpp_mv_loss.append(bpp_mv)
            bpp_mv_z_loss.append(bpp_mv_z)


        bpp=sum(rateloss)/6
        ms_ssim=sum(distortionloss)/6
        bpp_feature=sum(bpp_feature_loss)/6
        bpp_z=sum(bpp_z_loss)/6
        bpp_mv=sum(bpp_mv_loss)/6
        bpp_mv_z=sum(bpp_mv_z_loss)/6

        distribution_loss = bpp
        distortion = 1-ms_ssim
        rd_loss = train_lambda * distortion + distribution_loss
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        rd_loss.backward()

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        # clip_gradient(optimizer, 0.5)
        optimizer.step()
        aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        aux_optimizer.step()

        if global_step % cal_step == 0:
            cal_cnt += 1

            loss_=rd_loss.detach()
            sumloss += loss_

            summsssim+=ms_ssim.detach()
            sumbpp += bpp.detach()
            sumbpp_feature += bpp_feature.detach()
            sumbpp_z += bpp_z.detach()
            sumbpp_mv += bpp_mv.detach()
            sumbpp_mv_z+=bpp_mv_z.detach()

        if (batch_idx % 10) == 0 and cal_cnt >0:
            log = 'Train epoch:{:02} [{:4}/{:4} ({:3.0f}%)]\t loss {:.6f}\t  MS_SSIM {:.6f}\t Bpp {:.6f}\t Bpp_res {:.6f}\t Bpp_res_z {:.6f}\t Bpp_mv {:.6f}\t Bpp_mv_z {:.6f}\t Aux loss {:.6f}\t lr {}\t'.format(epoch, batch_idx , tot_iter, 100. * batch_idx / (tot_iter), sumloss / cal_cnt, summsssim/cal_cnt, sumbpp/cal_cnt,sumbpp_feature/cal_cnt,sumbpp_z/cal_cnt,sumbpp_mv/cal_cnt,sumbpp_mv_z/cal_cnt, aux_loss.item(),optimizer.param_groups[0]['lr'])
            logger.info(log)
            # tb_logger.add_scalar('loss', out_criterion["loss"], global_step)
            # tb_logger.add_scalar('loss_mse', out_criterion["mse_loss"],global_step)
            # tb_logger.add_scalar('loss_bpp', out_criterion["bpp_loss"], global_step)
            # tb_logger.add_scalar('loss_bpp_motion', out_criterion["bpp_loss.motion"],global_step)
            # tb_logger.add_scalar('loss_bpp_residual', out_criterion["bpp_loss.residual"], global_step)
            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_feature = sumbpp_mv = sumbpp_z=sumbpp_mv_z = sumloss = summsssim= 0

        # if global_step>1101486:
        #     break

    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    logger.info(log)
    net.update()
    return global_step



if __name__ == "__main__":
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

    ###Build Model
    ###P-frame compression
    model = FVC_ssim()
    net = model.cuda()

    ###I-frame compression
    # net_i=mbt2018(quality=8,metric='mse',pretrained=True)
    # net_i=bmshj2018_hyperprior(quality=8,metric='ms-ssim',pretrained=True)
    net_i=cheng2020_anchor(quality=6,metric='mse',pretrained=True)

    for para in net_i.parameters():
        para.requires_grad = False
    net_i = net_i.cuda()
    net_i.eval()

    ###Optimizer
    optimizer, aux_optimizer = configure_optimizers(net, base_lr,aux_learning_rate)
    global onestage_train_dataset, test_dataset
    tb_logger = SummaryWriter('./events')

    ###Dataset
    dataset = testdata_path
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(256)])

    ### two frames datasets
    onestage_train_dataset = DataSet(traindata_path)
    ### seven frames datasets
    twostage_train_dataset = VideoFolder(dataset, rnd_interval=True, rnd_temp_order=True, split="train",transform=train_transforms, )

    ### load pretrain models
    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)

    stepoch = global_step // (onestage_train_dataset.__len__() // (gpu_per_batch))# * gpu_num))
    for epoch in range(stepoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step)
            break
        global_step= train(epoch, global_step)
        save_model(model, global_step)
        # testhevcA(global_step, net, ref_i_dir, testfull=True)
        # testhevcB(global_step, net, ref_i_dir, testfull=True)
        # testhevcC(global_step, net, ref_i_dir, testfull=True)
        # testhevcD(global_step, net, ref_i_dir, testfull=True)
        # testhevcE(global_step, net, ref_i_dir, testfull=True)
        # testuvg(global_step, net, ref_i_dir, testfull=True)
        # testmcl(global_step, net, ref_i_dir, testfull=True)
        # testvtl(global_step, net, ref_i_dir, testfull=True)
