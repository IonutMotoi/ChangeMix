import argparse
import logging
import os
import pprint
import random
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import wandb

from dataset.semicd import SemiCDDataset
from model.semseg.dpt import DPT
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Supervised baselines: Labeled Only (--supervision semi) and Fully Supervised (--supervision full)')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--backbone', type=str, help='Backbone model to use: either dinov2_base or dinov2_small')
parser.add_argument('--model', type=str, help='Model type to use: dpt or deeplabv3plus', default='dpt')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--supervision', choices=['semi', 'full'], default='semi',
                    help='semi: labeled split only (Labeled Only); full: whole training set (Fully Supervised)')
parser.add_argument('--wandb', action='store_true', help='log to Weights & Biases (off by default)')


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, cfg, multiplier=None):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:

            imgA = imgA.cuda()
            imgB = imgB.cuda()

            if multiplier is not None:
                ori_h, ori_w = imgA.shape[-2:]
                new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                imgA = F.interpolate(imgA, (new_h, new_w), mode='bilinear', align_corners=True)
                imgB = F.interpolate(imgB, (new_h, new_w), mode='bilinear', align_corners=True)

            pred = model(imgA, imgB)

            if multiplier is not None:
                pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)

            pred = pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    return iou_class, overall_acc


def evaluate_v2(model, loader, cfg, multiplier=None):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()
    tp_pixel = AverageMeter()
    fp_pixel = AverageMeter()
    fn_pixel = AverageMeter()

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:

            imgA = imgA.cuda()
            imgB = imgB.cuda()

            if multiplier is not None:
                ori_h, ori_w = imgA.shape[-2:]
                new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                imgA = F.interpolate(imgA, (new_h, new_w), mode='bilinear', align_corners=True)
                imgB = F.interpolate(imgB, (new_h, new_w), mode='bilinear', align_corners=True)

            pred = model(imgA, imgB)

            if multiplier is not None:
                pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)

            pred = pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())

            pred_cpu = pred.cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            valid = mask_cpu != 255
            pred_bin = pred_cpu == 1
            mask_bin = mask_cpu == 1
            tp = np.logical_and(pred_bin, mask_bin) & valid
            fp = np.logical_and(pred_bin, np.logical_not(mask_bin)) & valid
            fn = np.logical_and(np.logical_not(pred_bin), mask_bin) & valid
            tp_pixel.update(tp.sum())
            fp_pixel.update(fp.sum())
            fn_pixel.update(fn.sum())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    precision = tp_pixel.sum / (tp_pixel.sum + fp_pixel.sum + 1e-10) * 100.0
    recall = tp_pixel.sum / (tp_pixel.sum + fn_pixel.sum + 1e-10) * 100.0

    return iou_class, overall_acc, precision, recall


def main():
    args = parser.parse_args()

    set_random_seed(args.seed)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    if args.backbone:
        cfg['backbone'] = args.backbone
    if args.model:
        cfg['model'] = args.model
    if args.lr:
        cfg['lr'] = args.lr

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

    cfg['batch_size'] *= 2

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

    local_rank = int(os.environ["LOCAL_RANK"])


    if cfg['model'] == 'dpt':
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

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                        output_device=local_rank, find_unused_parameters=True)
    elif cfg['model'] == 'deeplabv3plus':
        model = DeepLabV3Plus()

        optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

        if rank == 0:
            logger.info('Total params: {:.1f}M'.format(count_params(model)))

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                        output_device=local_rank, find_unused_parameters=False)
    else:
        raise NotImplementedError(f"Model {cfg['model']} not implemented.")

    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)

    if args.supervision == 'full':
        trainset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], f"splits/{cfg['dataset']}/train.txt", nsample=None)
    else:
        n_upsampled = {'whu': 3000, 'levir': 3000}
        trainset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=n_upsampled[cfg['dataset']])
    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')
    testset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'test')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True, seed=args.seed)
    valsampler     = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False, seed=args.seed)
    testsampler    = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, seed=args.seed)

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4,
                               drop_last=True, sampler=trainsampler, worker_init_fn=seed_worker, generator=g_l)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=valsampler,
                           worker_init_fn=seed_worker, generator=g_v)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=testsampler,
                            worker_init_fn=seed_worker, generator=g_t)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best_iou, previous_best_acc = 0.0, 0.0
    best_pre, best_rec = 0.0, 0.0
    best_epoch = 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']
        best_pre = checkpoint.get('best_pre', 0.0)
        best_rec = checkpoint.get('best_rec', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Best Changed IoU: {:.2f}, Overall Accuracy: {:.2f} @epoch-{:}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc, best_epoch))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (imgA, imgB, mask) in enumerate(trainloader):

            imgA, imgB, mask = imgA.cuda(), imgB.cuda(), mask.cuda()

            pred = model(imgA, imgB)

            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.detach().item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            if cfg['model'] == 'dpt':
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.detach().item(), iters)
                # wandb train metrics
                wandb.log({
                    'batch_step': iters,
                    'train/loss_all': loss.detach().item(),
                })

            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        iou_class, overall_acc, precision, recall = evaluate_v2(model, valloader, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.2f}'.format(iou_class[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}'.format(iou_class[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}'.format(overall_acc))
            logger.info('***** Evaluation ***** >>>> Precision: {:.2f}'.format(precision))
            logger.info('***** Evaluation ***** >>>> Recall: {:.2f}\n'.format(recall))

            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)
            writer.add_scalar('eval/precision', precision, epoch)
            writer.add_scalar('eval/recall', recall, epoch)

            # wandb evaluation metrics
            wandb.log({
                'epoch': epoch,
                'eval/unchanged_IoU': iou_class[0],
                'eval/changed_IoU': iou_class[1],
                'eval/overall_accuracy': overall_acc,
                'eval/precision': precision,
                'eval/recall': recall
            })

        is_best = iou_class[1] >= previous_best_iou
        previous_best_iou = max(iou_class[1], previous_best_iou)
        if is_best:
            previous_best_acc = overall_acc
            best_epoch = epoch
            best_pre = precision
            best_rec = recall

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
                'best_pre': best_pre,
                'best_rec': best_rec,
                'best_epoch': best_epoch,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_weights.pth'))

    # Load best model weights and evaluate on test set
    if rank == 0:
        logger.info('\n***** Final Test Evaluation with Best Model *****')

    # Load the best model weights
    best_weights = torch.load(os.path.join(args.save_path, 'best_weights.pth'), map_location='cpu')
    model.load_state_dict(best_weights)

    if rank == 0:
        logger.info(f'Loaded best model weights from epoch {best_epoch}')

    # Evaluate on test set
    test_iou_class, test_overall_acc, test_precision, test_recall = evaluate_v2(model, testloader, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)

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

        wandb.summary['best_changed_iou'] = previous_best_iou
        wandb.summary['best_overall_accuracy'] = previous_best_acc
        wandb.summary['best_precision'] = best_pre
        wandb.summary['best_recall'] = best_rec
        wandb.summary['best_f1_score'] = best_f1_score
        wandb.summary['best_epoch'] = best_epoch

        # Add test results to wandb summary
        wandb.summary['test_unchanged_iou'] = test_iou_class[0]
        wandb.summary['test_changed_iou'] = test_iou_class[1]
        wandb.summary['test_overall_accuracy'] = test_overall_acc
        wandb.summary['test_precision'] = test_precision
        wandb.summary['test_recall'] = test_recall
        wandb.summary['test_f1_score'] = test_f1_score

        logger.info(f"Best Model (Epoch {best_epoch}):")
        logger.info(f"  Changed IoU: {previous_best_iou:.4f}")
        logger.info(f"  Overall Accuracy: {previous_best_acc:.4f}")
        logger.info(f"  Precision: {best_pre:.4f}")
        logger.info(f"  Recall: {best_rec:.4f}")
        logger.info(f"  F1 Score: {best_f1_score:.4f}")

        logger.info(f"Final Test Results:")
        logger.info(f"  Test Changed IoU: {test_iou_class[1]:.4f}")
        logger.info(f"  Test Overall Accuracy: {test_overall_acc:.4f}")
        logger.info(f"  Test Precision: {test_precision:.4f}")
        logger.info(f"  Test Recall: {test_recall:.4f}")
        logger.info(f"  Test F1 Score: {test_f1_score:.4f}")

        wandb.finish()


if __name__ == '__main__':
    main()
