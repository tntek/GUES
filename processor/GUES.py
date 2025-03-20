import logging
import torch
from utils.meter import AverageMeter
from torch.cuda import amp
import torchvision.transforms as T
from datasets.make_dataloader import source_target_train_collate_fn
from datasets.bases import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score,accuracy_score

def generate_new_dataset(cfg, logger, s_dataset, t_dataset):
    train_set = []
    new_target_knnidx = []
    new_targetidx = []
    source_dataset = s_dataset.dataset.dataset
    target_dataset = t_dataset.dataset.dataset
    # combine_target_sample:
    for _, data in enumerate(tqdm(target_dataset)):
        t_img_path, vid, _, _,t_idx = data
        source_data = source_dataset[t_idx]
        s_img_path, _, camid, trackid, _  = source_data
        label = vid
        new_targetidx.append(t_idx)
        new_target_knnidx.append(t_idx)
        train_set.append(((s_img_path, t_img_path), (label, label), camid, trackid, (t_idx, t_idx)))

    logger.info('target match accuracy') 

    train_transforms = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    new_dataset = ImageDataset(train_set, train_transforms)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_loader = DataLoader(
            new_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True, drop_last = False, 
            num_workers=num_workers, collate_fn=source_target_train_collate_fn,
            pin_memory=True, persistent_workers=True,
            )

    return train_loader


def do_train_uda(cfg,
             model,vae_model,
             train_loader1,
             train_loader2,
             optimizer,
             scheduler
             ):

    log_period = cfg.SOLVER.LOG_PERIOD
    device = "cuda"
    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')


    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss3_meter = AverageMeter()
    acc_meter = AverageMeter()
    q_kappa_meter = AverageMeter()
    epoch = 1
    loss1_meter.reset()
    loss2_meter.reset()
    loss3_meter.reset()
    acc_meter.reset()
    q_kappa_meter.reset()
    scheduler.step(epoch)

    model.eval()
    vae_model.eval()
    train_loader = generate_new_dataset(cfg, logger, train_loader1, train_loader2)

    output_list = []
    label_list = []
    vp_bank = torch.randn(len(train_loader.dataset), 3,224,224).cuda()

    for n_iter, (imgs, vid, _, _, _, _) in enumerate(train_loader):
        vae_model.train()
        img = imgs[0].to(device)
        t_img = imgs[1].to(device) #target img
        t_pseudo_target = vid[0]

        tta_steps = 1
        for j in range(tta_steps):
            with amp.autocast(enabled=True):
                _, recon_loss, kld,_,_ = vae_model(t_img,img)
                loss1 = cfg.SOLVER.PAR*recon_loss
                loss2 = cfg.SOLVER.PAR_KL*kld
                classifier_loss = loss1 + loss2
            optimizer.zero_grad()
            classifier_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 20000)
            optimizer.step()

        with torch.no_grad():
            recon_x_new = vae_model(t_img,t_img,cal_loss=False)
            probs_new = model(**dict(x = recon_x_new, x2 = recon_x_new, return_feat_prob=True))
            recon_pre_new, _, _ = probs_new[1]
            acc = (recon_pre_new.max(1)[1] == t_pseudo_target.cuda()).float().mean()
            output_list.append(recon_pre_new.cpu())
            label_list.append(t_pseudo_target.cpu())

        loss1_meter.update(loss1.item(), img.shape[0])
        loss2_meter.update(loss2.item(), img.shape[0])
        loss3_meter.update(loss2.item(), img.shape[0])
        acc_meter.update(acc, img.shape[0])

        torch.cuda.synchronize()
        if (n_iter + 1) % log_period == 0 or (n_iter+1)==len(train_loader):
            logger.info("Epoch[{}] Iteration[{}/{}] Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f}, Acc: {:.5f}, Base Lr: {:.2e}"
                        .format(epoch, (n_iter + 1), len(train_loader),
                                loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
    output_all = torch.cat(output_list)
    label_all = torch.cat(label_list)
    q_kappa = cohen_kappa_score(label_all, output_all.max(1)[1].cpu().numpy(),weights='quadratic') 
    acc = accuracy_score(label_all,output_all.max(1)[1].cpu().numpy())
    logger.info("Epoch[{}] Iteration[{}/{}] q_kappa: {:.5f}, acc:{:.5f}"
                .format(epoch, (n_iter + 1), len(train_loader),q_kappa,acc))