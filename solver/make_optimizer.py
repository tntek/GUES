import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        # if 'pos_embed' in key or 'cls_token' in key or 'cam_embed' in key:
        #     print(key, 'key with no weight decay')
        #     weight_decay = 0.
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # for key, value in adv_net.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = cfg.SOLVER.BASE_LR
    #     weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #     if "bias" in key:
    #         lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    #         weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
    #     if cfg.SOLVER.LARGE_FC_LR:
    #         if "classifier" in key or "arcface" in key:
    #             lr = cfg.SOLVER.BASE_LR * 2
    #             print('Using two times learning rate for fc ')
    #     # if 'pos_embed' in key or 'cls_token' in key or 'cam_embed' in key:
    #     #     print(key, 'key with no weight decay')
    #     #     weight_decay = 0.
    #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    # else:
    #     optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    # optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer

# def make_optimizer_vae(cfg, model, center_criterion):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#         if cfg.SOLVER.LARGE_FC_LR:
#             if "classifier" in key or "arcface" in key:
#                 lr = cfg.SOLVER.BASE_LR * 2
#                 print('Using two times learning rate for fc ')
#         # if 'pos_embed' in key or 'cls_token' in key or 'cam_embed' in key:
#         #     print(key, 'key with no weight decay')
#         #     weight_decay = 0.
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

#     if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
#     elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
#         optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
#     optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

#     return optimizer, optimizer_center