import argparse
from copy import deepcopy
import logging
import os
import pprint
import random
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import wandb

from dataset.semicd import SemiCDDataset, SemiCDDatasetChangeMix
from model.semseg.dpt import DPT
from supervised import evaluate_v2
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from changemix import InstanceBank, paste_instances


parser = argparse.ArgumentParser(description='FixMatch (EMA teacher) for semi-supervised change detection, with ChangeMix')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', type=int)
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

# ChangeMix hyperparameters (optional overrides of the `changemix` section of the config)
parser.add_argument('--bank_size', type=int, help='Instance Bank capacity B')
parser.add_argument('--min_conf_thresh', type=float, help='minimum mean confidence tau_min of a pseudo-labeled instance')
parser.add_argument('--max_conf_thresh', type=float, help='maximum mean confidence tau_max of a pseudo-labeled instance')
parser.add_argument('--min_instance_size', type=int, help='minimum bounding-box side s_min (pixels)')
parser.add_argument('--max_instance_size', type=int, help='maximum bounding-box side s_max (pixels)')
parser.add_argument('--padding', type=int, help='context padding p around each instance (pixels)')
parser.add_argument('--instances_per_image', type=int, help='maximum number N of instances pasted per image')
parser.add_argument('--paste_attempts', type=int, help='placement attempts L per instance')
parser.add_argument('--paste_prob', type=float, help='probability p_mix of applying ChangeMix to a strong view')
parser.add_argument('--ema_ratio', type=float, default=0.996, help='Upper bound for the EMA decay rate for the teacher model')
parser.add_argument('--backbone', type=str, help='Backbone model to use: either dinov2_base or dinov2_small')
parser.add_argument('--only_sup_ratio', type=float, default=0.0, help='Fraction of total iterations trained with only the supervised loss (e.g. 0.1 = first 10%% of training)')
parser.add_argument('--wandb', action='store_true', help='log to Weights & Biases (off by default)')


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parser.parse_args()

    set_random_seed(args.seed)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # Command-line overrides of the ChangeMix hyperparameters
    for key in ['bank_size', 'min_conf_thresh', 'max_conf_thresh', 'min_instance_size', 'max_instance_size',
                'padding', 'instances_per_image', 'paste_attempts', 'paste_prob']:
        if getattr(args, key) is not None:
            cfg['changemix'][key] = getattr(args, key)
    if args.backbone:
        cfg['backbone'] = args.backbone
    cfg_cm = cfg['changemix']

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g_l = torch.Generator().manual_seed(args.seed + 10 + rank*1000)
    g_u = torch.Generator().manual_seed(args.seed + 20 + rank*1000)
    g_v = torch.Generator().manual_seed(args.seed + 30 + rank*1000)
    g_t = torch.Generator().manual_seed(args.seed + 40 + rank*1000)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}

        writer = SummaryWriter(args.save_path)

        # Weights & Biases logging is optional (--wandb); TensorBoard logs are always written to --save-path
        wandb.init(
            project=cfg.get('wandb_project', 'ChangeMix'),
            config=all_args,
            dir=args.save_path,
            mode='online' if args.wandb else 'disabled',
        )
        wandb.define_metric("train/*", step_metric="batch_step")
        wandb.define_metric("eval/*", step_metric="epoch")

        logger.info('{}\n'.format(pprint.pformat(all_args)))

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)

    if cfg['lock_backbone']:
        model.lock_backbone()

    optimizer = AdamW([{'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}],
                        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)

    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)

    trainset_u = SemiCDDatasetChangeMix(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiCDDatasetChangeMix(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')
    testset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'test')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l, shuffle=True, seed=args.seed)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u, shuffle=True, seed=args.seed)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False, seed=args.seed)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, seed=args.seed)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4,
                               drop_last=True, sampler=trainsampler_l, worker_init_fn=seed_worker, generator=g_l)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4,
                               drop_last=True, sampler=trainsampler_u, worker_init_fn=seed_worker, generator=g_u)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=valsampler,
                           worker_init_fn=seed_worker, generator=g_v)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=testsampler,
                            worker_init_fn=seed_worker, generator=g_t)

    total_iters = len(trainloader_u) * cfg['epochs']
    only_sup_iters = int(args.only_sup_ratio * total_iters)
    best_iou, best_acc = 0.0, 0.0
    best_ema_iou, best_ema_acc = 0.0, 0.0
    best_pre, best_rec = 0.0, 0.0
    best_ema_pre, best_ema_rec = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    instance_bank = InstanceBank(bank_size=cfg_cm['bank_size'],
                                 min_confidence_threshold=cfg_cm['min_conf_thresh'],
                                 max_confidence_threshold=cfg_cm['max_conf_thresh'],
                                 min_instance_size=cfg_cm['min_instance_size'],
                                 max_instance_size=cfg_cm['max_instance_size'],
                                 padding=cfg_cm['padding'])

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']
        best_acc = checkpoint['best_acc']
        best_ema_iou = checkpoint['best_ema_iou']
        best_ema_acc = checkpoint['best_ema_acc']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        best_pre = checkpoint.get('best_pre', 0.0)
        best_rec = checkpoint.get('best_rec', 0.0)
        best_ema_pre = checkpoint.get('best_ema_pre', 0.0)
        best_ema_rec = checkpoint.get('best_ema_rec', 0.0)

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Best Changed IoU: {:.2f}, Overall Accuracy: {:.2f} @epoch-{:}, EMA: Best Changed IoU: {:.2f}, Overall Accuracy: {:.2f} @epoch-{:}'.format(
                epoch, optimizer.param_groups[0]['lr'], best_iou, best_acc, best_epoch, best_ema_iou, best_ema_acc, best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, ((imgA_x, imgB_x, mask_x, imgA_x_s, imgB_x_s),
                (imgA_u_w, imgB_u_w, imgA_u_s, imgB_u_s,
                 _, _, ignore_mask, _, _)) in enumerate(loader):

            imgA_x, imgB_x, mask_x = imgA_x.cuda(), imgB_x.cuda(), mask_x.cuda()
            imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
            imgA_u_s, imgB_u_s = imgA_u_s.cuda(), imgB_u_s.cuda()
            ignore_mask = ignore_mask.cuda()

            iters = epoch * len(trainloader_u) + i

            with torch.no_grad():
                pred_u_w = model_ema(imgA_u_w, imgB_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            # ChangeMix, instance extraction: ground-truth instances of the labeled batch and pseudo-labeled
            # instances of the unlabeled batch, each taken from the weak or the strong view with probability 0.5
            use_augmented_labeled = random.random() < cfg_cm['prob_augment_labeled']
            imgA_lab, imgB_lab = (imgA_x_s.cuda(), imgB_x_s.cuda()) if use_augmented_labeled else (imgA_x, imgB_x)
            instance_bank.update(imgA_lab, imgB_lab, mask_x, torch.ones_like(mask_x).float(), is_labeled=True)

            if iters >= only_sup_iters:
                use_augmented_unlabeled = random.random() < cfg_cm['prob_augment_unlabeled']
                imgA_unlab, imgB_unlab = (imgA_u_s, imgB_u_s) if use_augmented_unlabeled else (imgA_u_w, imgB_u_w)
                instance_bank.update(imgA_unlab, imgB_unlab, mask_u_w, conf_u_w, is_labeled=False)

            mask_u_s, conf_u_s = mask_u_w.clone(), conf_u_w.clone()

            # ChangeMix, pasting: with probability paste_prob, instances from the bank are pasted into the strong
            # view, replacing the image content, the pseudo-label and the confidence of the pasted windows
            if random.random() < cfg_cm['paste_prob']:
                imgA_u_s, imgB_u_s, mask_u_s, conf_u_s = paste_instances(imgA_u_s, imgB_u_s, mask_u_s, conf_u_s, instance_bank,
                                instances_per_image=cfg_cm['instances_per_image'], paste_attempts=cfg_cm['paste_attempts'])

            model.train()

            pred_x = model(imgA_x, imgB_x)
            pred_u_s = model(imgA_u_s, imgB_u_s)

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s = criterion_u(pred_u_s, mask_u_s)
            loss_u_s = loss_u_s * ((conf_u_s >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_s = loss_u_s.sum() / (ignore_mask != 255).sum().item()

            # Supervised cold start: no unsupervised loss during the first only_sup_iters iterations
            unsup_weight = 0.0 if iters < only_sup_iters else 1.0

            loss = loss_x + unsup_weight * loss_u_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.detach().item())
            total_loss_x.update(loss_x.detach().item())
            total_loss_s.update(loss_u_s.detach().item())

            mask_ratio = (
                ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().float()
                / (ignore_mask != 255).sum().float()
            ).item()
            total_mask_ratio.update(mask_ratio)

            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), args.ema_ratio)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.detach().item(), iters)
                writer.add_scalar('train/loss_x', loss_x.detach().item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.detach().item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                # wandb train metrics
                wandb.log({
                    'batch_step': iters,
                    'train/loss_all': loss.detach().item(),
                    'train/loss_x': loss_x.detach().item(),
                    'train/loss_s': loss_u_s.detach().item(),
                    'train/mask_ratio': mask_ratio,
                })

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                                            total_loss_s.avg, total_mask_ratio.avg))

        iou_class, overall_acc, precision, recall = evaluate_v2(model, valloader, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)
        iou_class_ema, overall_acc_ema, precision_ema, recall_ema = evaluate_v2(model_ema, valloader, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.2f}, EMA: {:.2f}'.format(iou_class[0], iou_class_ema[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}: EMA: {:.2f}'.format(iou_class[1], iou_class_ema[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}, EMA: {:.2f}\n'.format(overall_acc, overall_acc_ema))
            logger.info('***** Evaluation ***** >>>> Precision: {:.2f}, EMA: {:.2f}'.format(precision, precision_ema))
            logger.info('***** Evaluation ***** >>>> Recall: {:.2f}, EMA: {:.2f}\n'.format(recall, recall_ema))

            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)

            writer.add_scalar('eval/unchanged_IoU_ema', iou_class_ema[0], epoch)
            writer.add_scalar('eval/changed_IoU_ema', iou_class_ema[1], epoch)
            writer.add_scalar('eval/overall_accuracy_ema', overall_acc_ema, epoch)

            writer.add_scalar('eval/precision', precision, epoch)
            writer.add_scalar('eval/recall', recall, epoch)
            writer.add_scalar('eval/precision_ema', precision_ema, epoch)
            writer.add_scalar('eval/recall_ema', recall_ema, epoch)

            # wandb evaluation metrics
            wandb.log({
                'epoch': epoch,
                'eval/unchanged_IoU': iou_class[0],
                'eval/changed_IoU': iou_class[1],
                'eval/overall_accuracy': overall_acc,
                'eval/unchanged_IoU_ema': iou_class_ema[0],
                'eval/changed_IoU_ema': iou_class_ema[1],
                'eval/overall_accuracy_ema': overall_acc_ema,
                'eval/precision': precision,
                'eval/recall': recall,
                'eval/precision_ema': precision_ema,
                'eval/recall_ema': recall_ema
            })

        is_best = iou_class[1] >= best_iou
        if is_best:
            best_iou = iou_class[1]
            best_acc = overall_acc
            best_epoch = epoch
            best_pre = precision
            best_rec = recall

        is_best_ema = iou_class_ema[1] >= best_ema_iou
        if is_best_ema:
            best_ema_iou = iou_class_ema[1]
            best_ema_acc = overall_acc_ema
            best_epoch_ema = epoch
            best_ema_pre = precision_ema
            best_ema_rec = recall_ema

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'best_acc': best_acc,
                'best_ema_iou': best_ema_iou,
                'best_ema_acc': best_ema_acc,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema,
                'best_pre': best_pre,
                'best_rec': best_rec,
                'best_ema_pre': best_ema_pre,
                'best_ema_rec': best_ema_rec
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_weights.pth'))
            if is_best_ema:
                torch.save(model_ema.state_dict(), os.path.join(args.save_path, 'best_ema_weights.pth'))

    # Load best EMA model weights and evaluate on test set
    if rank == 0:
        logger.info('\n***** Final Test Evaluation with Best EMA Model *****')

    # Load the best EMA model weights
    best_ema_weights = torch.load(os.path.join(args.save_path, 'best_ema_weights.pth'), map_location='cpu')
    model_ema.load_state_dict(best_ema_weights)

    if rank == 0:
        logger.info(f'Loaded best EMA model weights from epoch {best_epoch_ema}')

    # Evaluate on test set
    test_iou_class, test_overall_acc, test_precision, test_recall = evaluate_v2(model_ema, testloader, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)

    if rank == 0:
        test_f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-8) if (test_precision + test_recall) > 0 else 0.0

        logger.info('***** Final Test Results *****')
        logger.info('***** Test ***** >>>> Unchanged IoU: {:.2f}'.format(test_iou_class[0]))
        logger.info('***** Test ***** >>>> Changed IoU: {:.2f}'.format(test_iou_class[1]))
        logger.info('***** Test ***** >>>> Overall Accuracy: {:.2f}'.format(test_overall_acc))
        logger.info('***** Test ***** >>>> Precision: {:.2f}'.format(test_precision))
        logger.info('***** Test ***** >>>> Recall: {:.2f}'.format(test_recall))
        logger.info('***** Test ***** >>>> F1 Score: {:.2f}\n'.format(test_f1_score))

    if rank == 0:
        if (best_pre + best_rec) == 0:
            best_f1_score = 0.0
        else:
            best_f1_score = 2 * (best_pre * best_rec) / (best_pre + best_rec + 1e-8)

        if (best_ema_pre + best_ema_rec) == 0:
            best_ema_f1_score = 0.0
        else:
            best_ema_f1_score = 2 * (best_ema_pre * best_ema_rec) / \
                                (best_ema_pre + best_ema_rec + 1e-8)

        wandb.summary['best_changed_iou'] = best_iou
        wandb.summary['best_overall_accuracy'] = best_acc
        wandb.summary['best_precision'] = best_pre
        wandb.summary['best_recall'] = best_rec
        wandb.summary['best_f1_score'] = best_f1_score
        wandb.summary['best_epoch'] = best_epoch

        wandb.summary['best_ema_changed_iou'] = best_ema_iou
        wandb.summary['best_ema_overall_accuracy'] = best_ema_acc
        wandb.summary['best_ema_precision'] = best_ema_pre
        wandb.summary['best_ema_recall'] = best_ema_rec
        wandb.summary['best_ema_f1_score'] = best_ema_f1_score
        wandb.summary['best_epoch_ema'] = best_epoch_ema

        wandb.summary['test_unchanged_iou'] = test_iou_class[0]
        wandb.summary['test_changed_iou'] = test_iou_class[1]
        wandb.summary['test_overall_accuracy'] = test_overall_acc
        wandb.summary['test_precision'] = test_precision
        wandb.summary['test_recall'] = test_recall
        wandb.summary['test_f1_score'] = test_f1_score

        logger.info(f"Best Model (Epoch {best_epoch}):")
        logger.info(f"  Changed IoU: {best_iou:.4f}")
        logger.info(f"  Overall Accuracy: {best_acc:.4f}")
        logger.info(f"  Precision: {best_pre:.4f}")
        logger.info(f"  Recall: {best_rec:.4f}")
        logger.info(f"  F1 Score: {best_f1_score:.4f}")

        logger.info(f"Best EMA Model (Epoch {best_epoch_ema}):")
        logger.info(f"  Changed IoU: {best_ema_iou:.4f}")
        logger.info(f"  Overall Accuracy: {best_ema_acc:.4f}")
        logger.info(f"  Precision: {best_ema_pre:.4f}")
        logger.info(f"  Recall: {best_ema_rec:.4f}")
        logger.info(f"  F1 Score: {best_ema_f1_score:.4f}")

        logger.info(f"Final Test Results:")
        logger.info(f"  Test Changed IoU: {test_iou_class[1]:.4f}")
        logger.info(f"  Test Overall Accuracy: {test_overall_acc:.4f}")
        logger.info(f"  Test Precision: {test_precision:.4f}")
        logger.info(f"  Test Recall: {test_recall:.4f}")
        logger.info(f"  Test F1 Score: {test_f1_score:.4f}")

        wandb.finish()


if __name__ == '__main__':
    main()
