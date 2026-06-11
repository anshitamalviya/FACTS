#cps with gan plus pseudo discriminator epoch to 45 percent...
#Supervised fuzzy boundary loss (FB loss) plus unlabeled FB loss (Weight percent for both - 1.0)
#ACDC all slices
#changing dataset to ACDC
#ISSUE: Performance drop (0.88→0.80) with more unlabeled data (1312 vs 396 slices)
#ROOT CAUSE: Same epoch length (~11 iter) but 4.5x more unlabeled data → each sample seen 4.5x less frequently
#This leads to less stable pseudo-labels early in training, degrading performance
#FIXES: 1) Reduce pseudo_disc_start_frac to 0.35 (earlier pseudo-label learning)
#       2) Increase confidence threshold to 0.90 (stricter filtering)
#       3) Consider EMA for pseudo-labels or reduce unlabeled pool to match original ratio

#Model 1 metrics : [array([0.87615203, 0.78965103, 1.5166535 , 0.342132  ]), array([0.86771106, 0.76889159, 1.37705662, 0.5420708 ]), array([0.92166006, 0.86015873, 3.49008382, 0.8566796 ])]
#Model 2 metrics : [array([0.88233506, 0.79540623, 1.50663321, 0.52731783]), array([0.86806886, 0.76903107, 2.96914554, 0.64727138]), array([0.9245088 , 0.86360893, 3.62016441, 0.70540735])]
#Overall average metrics : [array([0.87924355, 0.79252863, 1.51164336, 0.43472491]), array([0.86788996, 0.76896133, 2.17310108, 0.59467109]), array([0.92308443, 0.86188383, 3.55512412, 0.78104348])]
#Average overall metric: [0.89007265 0.80779126 2.41328952 0.60347983]

#21 labeled_num
#Model 1 metrics : [array([0.90400613, 0.83088226, 1.3062217 , 0.39515458]), array([0.88930181, 0.8023431 , 1.14657006, 0.4120039 ]), array([0.94222052, 0.89281192, 2.01225539, 0.40870975])] 
#Model 2 metrics : [array([0.89917227, 0.8242151 , 1.39637837, 0.37090771]), array([0.89106216, 0.80497843, 1.87506282, 0.78417012]), array([0.93874325, 0.88827024, 3.44877077, 1.02522459])] 
#Overall average metrics : [array([0.9015892 , 0.82754868, 1.35130004, 0.38303115]), array([0.89018199, 0.80366076, 1.51081644, 0.59808701]), array([0.94048189, 0.89054108, 2.73051308, 0.71696717])]
#Average overall metric: [0.91075102 0.84058351 1.86420985 0.56602844]
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from networks.unet import PatchDiscriminator
from scipy.ndimage import distance_transform_edt, zoom
from medpy import metric
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Pseudo_Supervision_FB_loss',help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=136,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--adv_weight', type=float, default=0.01, help='weight for adversarial loss')
parser.add_argument('--pseudo_disc_start_epoch', type=int, default=-1,
                    help='epoch after which GT is not used for D; alternate fake1/fake2 as real/fake. If <0, computed from fraction')
parser.add_argument('--pseudo_disc_start_frac', type=float, default=0.35,
                    help='fraction of total epochs to use GT for D before alternating (used when start_epoch < 0). Lower = earlier pseudo-label learning')
parser.add_argument('--fb_loss_sigma', type=float, default=4.0, help='FB loss Gaussian band width (pixels)')
parser.add_argument('--fb_loss_wH', type=float, default=0.5, help='FB loss entropy weight')
parser.add_argument('--fb_loss_lambda', type=float, default=1.0, help='Weight for FB loss in supervised loss')
parser.add_argument('--fb_loss_eps', type=float, default=1e-6, help='Small epsilon for numerical stability')
parser.add_argument('--unl_fb_loss_lambda', type=float, default=1.0, help='Weight for FB loss on unlabeled data')
parser.add_argument('--confidence_threshold', type=float, default=0.90, help='Confidence threshold for pseudo-labeling. Higher = stricter filtering (needed with more unlabeled data)')
args = parser.parse_args()

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]
def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

bce_loss = nn.BCELoss()
def discriminator_loss(pred_real, pred_fake):
    real_labels = torch.ones_like(pred_real)
    fake_labels = torch.zeros_like(pred_fake)
    loss_real = bce_loss(pred_real, real_labels)
    loss_fake = bce_loss(pred_fake, fake_labels)
    return (loss_real + loss_fake) * 0.5

def generator_adv_loss(pred_fake):
    real_labels = torch.ones_like(pred_fake)
    return bce_loss(pred_fake, real_labels)

def compute_boundary_distance_one_vs_all(gt_onehot_np):
    """Compute distance-to-boundary per class for one-vs-all GT masks.

    Args:
        gt_onehot_np: numpy array shape [C, H, W] with binary {0,1} per class.
    Returns:
        dist_np: numpy array shape [C, H, W] with non-negative distances.
    """
    C, H, W = gt_onehot_np.shape
    dist_list = []
    for c in range(C):
        mask = gt_onehot_np[c].astype(np.bool_)
        # d_k^fg: Euclidean distance to nearest foreground pixel (N_k=1)
        d_fg = distance_transform_edt(mask == 0)
        # d_k^bg: Euclidean distance to nearest background pixel (N_k=0)
        d_bg = distance_transform_edt(mask == 1)
        # b_k(p) = N_k(p)*d_k^bg(p) + (1-N_k(p))*d_k^fg(p): distance to opposite region
        dist = np.where(mask, d_bg, d_fg).astype(np.float32)
        dist_list.append(dist)
    dist_np = np.stack(dist_list, axis=0)
    return dist_np

def fb_loss_multiclass(outputs_soft, labels_long, sigma_px, w_H, eps):
    """FB loss multiclass loss per batch.

    Args:
        outputs_soft: torch.FloatTensor [B, C, H, W], softmax probabilities.
        labels_long: torch.LongTensor [B, H, W], ground-truth labels.
        sigma_px: float, Gaussian width in pixels.
        w_H: float in [0,1], entropy vs margin blend weight.
        eps: small float for numerical stability.
    Returns:
        loss: torch scalar, averaged over batch and classes.
    """
    device = outputs_soft.device
    B, C, H, W = outputs_soft.shape

    # One-hot labels for BCE targets
    y_onehot = torch.nn.functional.one_hot(labels_long, num_classes=C).permute(0, 3, 1, 2).float()  # [B,C,H,W]

    # Compute classwise boundary distances per sample on CPU via EDT, then back to device
    dist_maps = []
    labels_cpu = y_onehot.detach().cpu().numpy()
    for b in range(B):
        dist_np = compute_boundary_distance_one_vs_all(labels_cpu[b])  # [C,H,W]
        dist_maps.append(torch.from_numpy(dist_np))
    d_bnd = torch.stack(dist_maps, dim=0).to(device=device, dtype=outputs_soft.dtype)  # [B,C,H,W]

    # Boundary band phi(c,x)
    denom = 2.0 * (sigma_px ** 2) + 1e-12
    phi = torch.exp(- (d_bnd * d_bnd) / denom)  # [B,C,H,W], in [0,1]

    # Uncertainty gate U(x): blend of multiclass entropy and margin proxy
    p = outputs_soft.clamp(min=1e-6, max=1.0 - 1e-6)
    log_p = torch.log(p)
    H_mc = - (p * log_p).sum(dim=1, keepdim=True) / np.log(C)  # [B,1,H,W], in [0,1]
    M_mc = 1.0 - p.max(dim=1, keepdim=True)[0]                  # [B,1,H,W], in [0,1]
    #U = w_H * H_mc + (1.0 - w_H) * M_mc                         # [B,1,H,W]
    U = H_mc

    # Fuzzy weight mu(c,x) and detach as recommended
    mu = (phi * U).detach()  # [B,C,H,W]

    # Per-class BCE
    bce_pos = -(y_onehot * torch.log(p))
    bce_neg = -((1.0 - y_onehot) * torch.log(1.0 - p))
    bce = bce_pos + bce_neg  # [B,C,H,W]

    # Weighted, normalized BCE per class then mean over classes
    num = (mu * bce).flatten(2).sum(dim=2)            # [B,C]
    den = mu.flatten(2).sum(dim=2) + eps               # [B,C]
    per_class = num / den                              # [B,C]
    loss = per_class.mean()                            # scalar
    return loss


def fb_loss_multiclass_masked(outputs_soft, labels_long, sigma_px, w_H, eps, mask=None):
    """FB loss multiclass loss with optional spatial mask.

    Args:
        outputs_soft: torch.FloatTensor [B, C, H, W], softmax probabilities.
        labels_long: torch.LongTensor [B, H, W], supervising labels (e.g., pseudo labels).
        sigma_px: float, Gaussian width in pixels.
        w_H: float in [0,1], entropy vs margin blend weight.
        eps: small float for numerical stability.
        mask: optional torch.BoolTensor [B, H, W] selecting confident pixels.
    Returns:
        loss: torch scalar.
    """
    device = outputs_soft.device
    B, C, H, W = outputs_soft.shape

    y_onehot = torch.nn.functional.one_hot(labels_long, num_classes=C).permute(0, 3, 1, 2).float()

    # distance maps from labels_long (pseudo labels allowed)
    dist_maps = []
    labels_cpu = y_onehot.detach().cpu().numpy()
    for b in range(B):
        dist_np = compute_boundary_distance_one_vs_all(labels_cpu[b])
        dist_maps.append(torch.from_numpy(dist_np))
    d_bnd = torch.stack(dist_maps, dim=0).to(device=device, dtype=outputs_soft.dtype)

    denom = 2.0 * (sigma_px ** 2) + 1e-12
    phi = torch.exp(- (d_bnd * d_bnd) / denom)

    p = outputs_soft.clamp(min=1e-6, max=1.0 - 1e-6)
    log_p = torch.log(p)
    H_mc = - (p * log_p).sum(dim=1, keepdim=True) / np.log(C)
    # Use entropy only as in supervised version above
    U = H_mc

    mu = (phi * U).detach()

    if mask is not None:
        # Broadcast mask to [B, C, H, W]
        mask_bc = mask.unsqueeze(1).expand(-1, C, -1, -1).to(device=device, dtype=mu.dtype)
        mu = mu * mask_bc
        # Also avoid supervising negative class BCE outside mask by zeroing both numerator and denominator via mu

    bce_pos = -(y_onehot * torch.log(p))
    bce_neg = -((1.0 - y_onehot) * torch.log(1.0 - p))
    bce = bce_pos + bce_neg

    num = (mu * bce).flatten(2).sum(dim=2)
    den = mu.flatten(2).sum(dim=2) + eps
    per_class = num / den
    loss = per_class.mean()
    return loss


def evaluate_training_subset(model, dataset_base_dir, indices, num_classes, device='cuda'):
    """Evaluate model on a subset of training data (labeled or unlabeled).
    
    Args:
        model: The model to evaluate
        dataset_base_dir: Base directory of the dataset
        indices: List of indices to evaluate on (corresponds to sample_list indices)
        num_classes: Number of classes
        device: Device to run evaluation on
    
    Returns:
        metric_array: Array of shape [num_classes-1, 2] where each row is [dice, hd95] for a class
    """
    model.eval()
    
    # Read the sample list to get file names
    with open(dataset_base_dir + "/train_slices.list", "r") as f1:
        sample_list = f1.readlines()
    sample_list = [item.replace("\n", "") for item in sample_list]
    
    # Initialize metric accumulators
    metric_accum = np.zeros((num_classes - 1, 2))  # [num_classes-1, dice/hd95]
    valid_samples = 0
    
    with torch.no_grad():
        for idx in indices:
            if idx >= len(sample_list):
                continue
                
            case = sample_list[idx]
            try:
                # Load data directly from HDF5 without transforms
                h5f = h5py.File(dataset_base_dir + "/data/slices/{}.h5".format(case), "r")
                image = h5f["image"][:]
                label = h5f["label"][:]
                h5f.close()
                
                # Convert to tensors
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                if isinstance(label, np.ndarray):
                    label = torch.from_numpy(label).long()
                
                # Add batch and channel dimensions if needed
                if image.dim() == 2:
                    image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif image.dim() == 3:
                    image = image.unsqueeze(0)  # [1, C, H, W]
                
                if label.dim() == 2:
                    label = label.unsqueeze(0)  # [1, H, W]
                
                # Resize to patch size if needed (for consistency with training)
                if image.shape[2] != 256 or image.shape[3] != 256:
                    # Resize image
                    img_np = image.squeeze(0).squeeze(0).numpy()
                    img_resized = zoom(img_np, (256.0 / img_np.shape[0], 256.0 / img_np.shape[1]), order=1)
                    image = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()
                    
                    # Resize label
                    label_np = label.squeeze(0).numpy()
                    label_resized = zoom(label_np, (256.0 / label_np.shape[0], 256.0 / label_np.shape[1]), order=0)
                    label = torch.from_numpy(label_resized).unsqueeze(0).long()
                
                image = image.to(device)
                
                # Forward pass
                output = model(image)
                prediction = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
                prediction = prediction.cpu().numpy()
                label_np = label.squeeze(0).numpy()
                
                # Compute metrics for each class
                for class_idx in range(1, num_classes):
                    pred_binary = (prediction == class_idx).astype(np.float32)
                    label_binary = (label_np == class_idx).astype(np.float32)
                    
                    if pred_binary.sum() > 0 or label_binary.sum() > 0:
                        try:
                            dice_val = metric.binary.dc(pred_binary, label_binary)
                            hd95_val = metric.binary.hd95(pred_binary, label_binary)
                        except:
                            dice_val = 0.0
                            hd95_val = 0.0
                    else:
                        dice_val = 0.0
                        hd95_val = 0.0
                    
                    metric_accum[class_idx - 1, 0] += dice_val
                    metric_accum[class_idx - 1, 1] += hd95_val
                
                valid_samples += 1
            except Exception as e:
                logging.warning(f"Error evaluating sample {idx}: {e}")
                continue
    
    # Average over all samples
    if valid_samples > 0:
        metric_array = metric_accum / valid_samples
    else:
        metric_array = np.zeros((num_classes - 1, 2))
    
    model.train()
    return metric_array


def train(args, snapshot_path):
    #torch.cuda.set_device(1)
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    #model1 = create_model()
    #model2 = create_model()
    # Create two UNet models with different initializations
    torch.manual_seed(args.seed)
    model1 = create_model()

    torch.manual_seed(args.seed + 100)  # Different seed for model2
    model2 = create_model()
    #def worker_init_fn(worker_id):
     #   random.seed(args.seed + worker_id)

    D = PatchDiscriminator(in_channels=args.num_classes).cuda()
    optimizerD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    # Determine pseudo discriminator start epoch
    if args.pseudo_disc_start_epoch is not None and args.pseudo_disc_start_epoch >= 0:
        pseudo_start_epoch = args.pseudo_disc_start_epoch
    else:
        pseudo_start_epoch = int(np.ceil(args.pseudo_disc_start_frac * max_epoch))
        
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1  = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

             # --- Discriminator forward ---
            with torch.no_grad():
                label_batch = label_batch.long()
                real_onehot_full = nn.functional.one_hot(label_batch, num_classes=args.num_classes)
                real_onehot_full = real_onehot_full.permute(0, 3, 1, 2).float()
                real_onehot = real_onehot_full[:args.labeled_bs]

            pred_fake1 = D(outputs_soft1.detach()[args.labeled_bs:])
            pred_fake2 = D(outputs_soft2.detach()[args.labeled_bs:])

            # --- Discriminator update ---
            optimizerD.zero_grad()
            # Before pseudo phase: use real GT as real; after: alternate fake1/fake2 roles
            if epoch_num < pseudo_start_epoch:
                pred_real = D(real_onehot)
                lossD = discriminator_loss(pred_real, pred_fake1) + discriminator_loss(pred_real, pred_fake2)
            else:
                # Alternate by iteration: even -> fake1 as real, fake2 as fake; odd -> swap
                if (iter_num % 2) == 0:
                    lossD = discriminator_loss(pred_fake1, pred_fake2)
                else:
                    lossD = discriminator_loss(pred_fake2, pred_fake1)
            lossD.backward()
            optimizerD.step()

            # Get confidence scores for pseudo-labeling
            confidence1 = torch.max(outputs_soft1[args.labeled_bs:], dim=1)[0]
            confidence2 = torch.max(outputs_soft2[args.labeled_bs:], dim=1)[0]

            # Create confidence masks (only use high-confidence predictions)
            # Higher threshold needed with more unlabeled data to maintain quality
            mask1 = confidence1 > args.confidence_threshold
            mask2 = confidence2 > args.confidence_threshold

            #loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
             #   outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            #loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
             #   outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            # Supervised loss on labeled subset only
            #loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
            #    outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            #loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
            #    outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            # Supervised losses on labeled subset only: Dice + lambda_FB * FB loss
            dice1 = dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            dice2 = dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            fb_loss1 = fb_loss_multiclass(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].long(), args.fb_loss_sigma, args.fb_loss_wH, args.fb_loss_eps)
            fb_loss2 = fb_loss_multiclass(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].long(), args.fb_loss_sigma, args.fb_loss_wH, args.fb_loss_eps)
            loss1 = dice1 + args.fb_loss_lambda * fb_loss1
            loss2 = dice2 + args.fb_loss_lambda * fb_loss2

            #pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            #pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            #pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
            #pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)

            # Cross-pseudo supervision
            pseudo_labels1 = torch.argmax(outputs_soft1.detach()[args.labeled_bs:], dim=1)
            pseudo_labels2 = torch.argmax(outputs_soft2.detach()[args.labeled_bs:], dim=1)

            # Model 2 teaches Model 1 (only high-confidence predictions)
            if mask2.any():
                # flatten spatial dims so we can mask on unlabeled subset
                unl_outputs1 = outputs1[args.labeled_bs:]
                B, C, H, W = unl_outputs1.shape
                logits = unl_outputs1.permute(0, 2, 3, 1).contiguous().view(-1, C)
                pseudo_flat = pseudo_labels2.view(-1)
                mask_flat = mask2.view(-1)
                masked_logits = logits[mask_flat]
                masked_pseudo = pseudo_flat[mask_flat]
                pseudo_supervision1 = ce_loss(masked_logits, masked_pseudo)
            else:
                pseudo_supervision1 = torch.tensor(0.0, device=outputs1.device)

            # Model 1 teaches Model 2 (only high-confidence predictions)
            if mask1.any():
                unl_outputs2 = outputs2[args.labeled_bs:]
                B, C, H, W = unl_outputs2.shape
                logits2 = unl_outputs2.permute(0, 2, 3, 1).contiguous().view(-1, C)
                pseudo1_flat = pseudo_labels1.view(-1)
                mask1_flat   = mask1.view(-1)
                masked_logits2 = logits2[mask1_flat]
                masked_pseudo1 = pseudo1_flat[mask1_flat]
                pseudo_supervision2 = ce_loss(masked_logits2, masked_pseudo1)
            else:
                pseudo_supervision2 = torch.tensor(0.0, device=outputs2.device)


            # Unlabeled FB loss using peer pseudo-labels and confidence masks
            if mask2.any():
                unl_fb_loss1 = fb_loss_multiclass_masked(
                    outputs_soft1[args.labeled_bs:],
                    pseudo_labels2,
                    args.fb_loss_sigma,
                    args.fb_loss_wH,
                    args.fb_loss_eps,
                    mask=mask2
                )
            else:
                unl_fb_loss1 = torch.tensor(0.0, device=outputs1.device)

            if mask1.any():
                unl_fb_loss2 = fb_loss_multiclass_masked(
                    outputs_soft2[args.labeled_bs:],
                    pseudo_labels1,
                    args.fb_loss_sigma,
                    args.fb_loss_wH,
                    args.fb_loss_eps,
                    mask=mask1
                )
            else:
                unl_fb_loss2 = torch.tensor(0.0, device=outputs2.device)

            # --- Adversarial Generator Loss ---
            adv1 = generator_adv_loss(D(outputs_soft1[args.labeled_bs:]))
            adv2 = generator_adv_loss(D(outputs_soft2[args.labeled_bs:]))

            model1_loss = (
                loss1
                + args.unl_fb_loss_lambda * unl_fb_loss1
                + consistency_weight * pseudo_supervision1
                + args.adv_weight * adv1
            )
            model2_loss = (
                loss2
                + args.unl_fb_loss_lambda * unl_fb_loss2
                + consistency_weight * pseudo_supervision2
                + args.adv_weight * adv2
            )

            loss = model1_loss + model2_loss


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/supervised_loss1', loss1, iter_num)
            writer.add_scalar('loss/supervised_loss2', loss2, iter_num)
            writer.add_scalar('loss/dice1', dice1, iter_num)
            writer.add_scalar('loss/dice2', dice2, iter_num)
            writer.add_scalar('loss/fb_loss1', fb_loss1, iter_num)
            writer.add_scalar('loss/fb_loss2', fb_loss2, iter_num)
            writer.add_scalar('loss/unl_fb_loss1', unl_fb_loss1, iter_num)
            writer.add_scalar('loss/unl_fb_loss2', unl_fb_loss2, iter_num)
            writer.add_scalar('loss/pseudo_supervision1', pseudo_supervision1, iter_num)
            writer.add_scalar('loss/pseudo_supervision2', pseudo_supervision2, iter_num)
            writer.add_scalar('loss/discriminator', lossD, iter_num)
            writer.add_scalar('loss/adv1', adv1, iter_num)
            writer.add_scalar('loss/adv2', adv2, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        
        # Evaluate on training data (labeled and unlabeled) and validation data after each epoch
        if epoch_num >= 0:  # Evaluate from epoch 0
            logging.info(f"Evaluating on training and validation data after epoch {epoch_num}...")
            
            # Sample a subset for evaluation to avoid it taking too long
            # Use up to 100 samples from labeled and unlabeled data
            eval_labeled_idxs = labeled_idxs[:min(100, len(labeled_idxs))]
            eval_unlabeled_idxs = unlabeled_idxs[:min(100, len(unlabeled_idxs))]
            
            # Evaluate Model 1 on labeled data
            model1.eval()
            labeled_metrics1 = evaluate_training_subset(
                model1, args.root_path, eval_labeled_idxs, num_classes, device='cuda')
            model1.train()
            
            # Evaluate Model 1 on unlabeled data
            model1.eval()
            unlabeled_metrics1 = evaluate_training_subset(
                model1, args.root_path, eval_unlabeled_idxs, num_classes, device='cuda')
            model1.train()
            
            # Evaluate Model 2 on labeled data
            model2.eval()
            labeled_metrics2 = evaluate_training_subset(
                model2, args.root_path, eval_labeled_idxs, num_classes, device='cuda')
            model2.train()
            
            # Evaluate Model 2 on unlabeled data
            model2.eval()
            unlabeled_metrics2 = evaluate_training_subset(
                model2, args.root_path, eval_unlabeled_idxs, num_classes, device='cuda')
            model2.train()
            
            # Evaluate on validation data
            # Model 1 validation
            model1.eval()
            val_metric_list1 = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(
                    sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                val_metric_list1 += np.array(metric_i)
            val_metric_list1 = val_metric_list1 / len(db_val)
            model1.train()
            
            # Model 2 validation
            model2.eval()
            val_metric_list2 = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(
                    sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                val_metric_list2 += np.array(metric_i)
            val_metric_list2 = val_metric_list2 / len(db_val)
            model2.train()
            
            # Log metrics for Model 1 - Labeled
            for class_i in range(num_classes - 1):
                writer.add_scalar(f'train_labeled/model1_class_{class_i+1}_dice', 
                                 labeled_metrics1[class_i, 0], epoch_num)
                writer.add_scalar(f'train_labeled/model1_class_{class_i+1}_hd95', 
                                 labeled_metrics1[class_i, 1], epoch_num)
            mean_dice_labeled1 = np.mean(labeled_metrics1[:, 0])
            mean_hd95_labeled1 = np.mean(labeled_metrics1[:, 1])
            writer.add_scalar('train_labeled/model1_mean_dice', mean_dice_labeled1, epoch_num)
            writer.add_scalar('train_labeled/model1_mean_hd95', mean_hd95_labeled1, epoch_num)
            
            # Log metrics for Model 1 - Unlabeled
            for class_i in range(num_classes - 1):
                writer.add_scalar(f'train_unlabeled/model1_class_{class_i+1}_dice', 
                                 unlabeled_metrics1[class_i, 0], epoch_num)
                writer.add_scalar(f'train_unlabeled/model1_class_{class_i+1}_hd95', 
                                 unlabeled_metrics1[class_i, 1], epoch_num)
            mean_dice_unlabeled1 = np.mean(unlabeled_metrics1[:, 0])
            mean_hd95_unlabeled1 = np.mean(unlabeled_metrics1[:, 1])
            writer.add_scalar('train_unlabeled/model1_mean_dice', mean_dice_unlabeled1, epoch_num)
            writer.add_scalar('train_unlabeled/model1_mean_hd95', mean_hd95_unlabeled1, epoch_num)
            
            # Log metrics for Model 2 - Labeled
            for class_i in range(num_classes - 1):
                writer.add_scalar(f'train_labeled/model2_class_{class_i+1}_dice', 
                                 labeled_metrics2[class_i, 0], epoch_num)
                writer.add_scalar(f'train_labeled/model2_class_{class_i+1}_hd95', 
                                 labeled_metrics2[class_i, 1], epoch_num)
            mean_dice_labeled2 = np.mean(labeled_metrics2[:, 0])
            mean_hd95_labeled2 = np.mean(labeled_metrics2[:, 1])
            writer.add_scalar('train_labeled/model2_mean_dice', mean_dice_labeled2, epoch_num)
            writer.add_scalar('train_labeled/model2_mean_hd95', mean_hd95_labeled2, epoch_num)
            
            # Log metrics for Model 2 - Unlabeled
            for class_i in range(num_classes - 1):
                writer.add_scalar(f'train_unlabeled/model2_class_{class_i+1}_dice', 
                                 unlabeled_metrics2[class_i, 0], epoch_num)
                writer.add_scalar(f'train_unlabeled/model2_class_{class_i+1}_hd95', 
                                 unlabeled_metrics2[class_i, 1], epoch_num)
            mean_dice_unlabeled2 = np.mean(unlabeled_metrics2[:, 0])
            mean_hd95_unlabeled2 = np.mean(unlabeled_metrics2[:, 1])
            writer.add_scalar('train_unlabeled/model2_mean_dice', mean_dice_unlabeled2, epoch_num)
            writer.add_scalar('train_unlabeled/model2_mean_hd95', mean_hd95_unlabeled2, epoch_num)
            
            # Log metrics for Model 1 - Validation
            for class_i in range(num_classes - 1):
                writer.add_scalar(f'val/model1_class_{class_i+1}_dice', 
                                 val_metric_list1[class_i, 0], epoch_num)
                writer.add_scalar(f'val/model1_class_{class_i+1}_hd95', 
                                 val_metric_list1[class_i, 1], epoch_num)
            mean_dice_val1 = np.mean(val_metric_list1[:, 0])
            mean_hd95_val1 = np.mean(val_metric_list1[:, 1])
            writer.add_scalar('val/model1_mean_dice', mean_dice_val1, epoch_num)
            writer.add_scalar('val/model1_mean_hd95', mean_hd95_val1, epoch_num)
            
            # Log metrics for Model 2 - Validation
            for class_i in range(num_classes - 1):
                writer.add_scalar(f'val/model2_class_{class_i+1}_dice', 
                                 val_metric_list2[class_i, 0], epoch_num)
                writer.add_scalar(f'val/model2_class_{class_i+1}_hd95', 
                                 val_metric_list2[class_i, 1], epoch_num)
            mean_dice_val2 = np.mean(val_metric_list2[:, 0])
            mean_hd95_val2 = np.mean(val_metric_list2[:, 1])
            writer.add_scalar('val/model2_mean_dice', mean_dice_val2, epoch_num)
            writer.add_scalar('val/model2_mean_hd95', mean_hd95_val2, epoch_num)
            
            # Log summary
            logging.info(f'Epoch {epoch_num} - Model1 Labeled: Dice={mean_dice_labeled1:.4f}, HD95={mean_hd95_labeled1:.4f}')
            logging.info(f'Epoch {epoch_num} - Model1 Unlabeled: Dice={mean_dice_unlabeled1:.4f}, HD95={mean_hd95_unlabeled1:.4f}')
            logging.info(f'Epoch {epoch_num} - Model1 Validation: Dice={mean_dice_val1:.4f}, HD95={mean_hd95_val1:.4f}')
            logging.info(f'Epoch {epoch_num} - Model2 Labeled: Dice={mean_dice_labeled2:.4f}, HD95={mean_hd95_labeled2:.4f}')
            logging.info(f'Epoch {epoch_num} - Model2 Unlabeled: Dice={mean_dice_unlabeled2:.4f}, HD95={mean_hd95_unlabeled2:.4f}')
            logging.info(f'Epoch {epoch_num} - Model2 Validation: Dice={mean_dice_val2:.4f}, HD95={mean_hd95_val2:.4f}')
        
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
