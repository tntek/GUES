import os
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import create_scheduler
from processor.GUES import do_train_uda
from vae_model import VAE
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from convnets import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/uda.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    else:
        pass

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.UDA_STAGE == 'UDA':
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    random_dim = 1024
    ndomains = 2
    class_num = 5
    encoder= Encoder_Conv_VAE_CELEBA(latent_dim=10)
    decoder = Decoder_Conv_AE_CELEBA(latent_dim=10).cuda()
    vae_model = VAE(encoder,decoder,reconstruction_loss='mse').cuda()
    model = make_model(cfg, num_class=num_classes).cuda()

    param_group = []
    for k, v in model.named_parameters():
        v.requires_grad = False

    model.eval()
    optimizer = torch.optim.SGD(vae_model.parameters(), cfg.SOLVER.BASE_LR,weight_decay=cfg.SOLVER.WEIGHT_DECAY,momentum = 0.9)
    scheduler = create_scheduler(cfg, optimizer)
    
    if cfg.MODEL.UDA_STAGE == 'UDA':
        do_train_uda(
        cfg,
        model,vae_model,
        train_loader1,
        train_loader2,
        optimizer,
        scheduler
    )