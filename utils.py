import torch.nn.functional as F
import torch
import os
import random
from PIL import Image
from torch.utils.data import IterableDataset

# Contrastive loss for SimCLR (ntXent loss)
def nt_xent_loss(z_i, z_j, labels=None, temperature=0.07):
    """
    Args:
        z_i: Tensor of shape [batch_size, feature_dim]
        z_j: Tensor of shape [batch_size, feature_dim]
        labels: Tensor of shape [batch_size], optional
        temperature: scalar float
    Returns:
        Scalar loss
    """
    device = z_i.device
    batch_size, feat_dim = z_i.shape

    # Stack views along dim=1 -> [batch_size, 2, feat_dim]
    features = torch.stack([z_i, z_j], dim=1)
    features = features.view(batch_size * 2, feat_dim)
    features = F.normalize(features, dim=1)

    # Create positive mask
    if labels is not None:
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    else:
        # If no labels, only positive pairs are different views of the same sample
        mask = torch.eye(batch_size, device=device)
        mask = mask.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

    # Mask out self-similarities on the diagonal
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size * 2, device=device)
    mask = mask * logits_mask

    # Compute similarity matrix scaled by temperature
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

    # Average log_prob over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    # Loss
    loss = -mean_log_prob_pos.mean()
    return loss

# Similarity loss
def similarity_loss(features_pos, temperature=0.07, 
                   contrast_mode='all', base_temperature=0.07):
    """
    Similarity Loss function.
    Adapted from https://github.com/HobbitLong/SupContrast
    
    Args:
        features_pos: Positive features tensor [bsz, n_views, feature_dim] or [bsz, feature_dim]
        temperature: Temperature scaling parameter
        contrast_mode: 'one' or 'all' - which features to use as anchors
        base_temperature: Base temperature for loss scaling

    Returns:
        Scalar loss value
    """

    # Handle 2D input by adding n_views dimension
    """if len(features_pos.shape) == 2:
        features_pos = features_pos.unsqueeze(1)"""  # [N, D] -> [N, 1, D]
    
    features = features_pos
    n_pos = features_pos.size(0)
    labels = torch.ones(n_pos, dtype=torch.int64)
    mask = None
    features = F.normalize(features, dim=-1)  # normalize embeddings
        
    device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
    if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device) #Computes element-wise equality
    else:
        mask = mask.float().to(device) #dimension [batch_size, batch_size]

    contrast_count = features.shape[1] #n_views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]#features: [bsz, n_views, ...], so only get n_views==0
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature) # matrix multiplication
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    
    logits = anchor_dot_contrast - logits_max.detach()
    #logits = anchor_dot_contrast

    mask = mask.repeat(anchor_count, contrast_count)#anchor_count==contrast_count==n_views==2, repeat(n_views,n_views)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,#the dimension along which to scatter
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),#index tensor, specifying which elements to modify
        0
    )
    
    mask = mask * logits_mask
        
    log_prob = logits
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs) # avoid division by 0
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs #dimension [batch_size * n_views]
    
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, n_pos).mean()

    return loss

# Batch creation to handle both positive and negative patches
class BalancedBatchDataset(IterableDataset):
    def __init__(self, positive_dir, negative_dir, transform_pos, transform_neg, batch_size):
        self.positive_paths = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.negative_paths = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform_pos = transform_pos
        self.transform_neg = transform_neg
        self.batch_size = batch_size
        self.r = len(self.positive_paths)/len(self.negative_paths)
        self.batch_size_pos = round((self.batch_size * self.r) / (self.r + 1))
        self.batch_size_neg = round((self.batch_size) / ( self.r + 1))
        self.num_batches = round((len(self.positive_paths) + len(self.negative_paths)) / self.batch_size)
       
    def __iter__(self):
        pos_paths = self.positive_paths.copy()
        neg_paths = self.negative_paths.copy()
        random.shuffle(pos_paths)
        random.shuffle(neg_paths)

        for i in range(self.num_batches):
            
            pos_batch = [pos_paths[(i + k) % len(pos_paths)] for k in range(self.batch_size_pos)]
            neg_batch = [neg_paths[(i + k) % len(neg_paths)] for k in range(self.batch_size_neg)]

            pos_aug1 = [self.transform_pos(Image.open(p).convert('RGB')) for p in pos_batch]
            pos_aug2 = [self.transform_pos(Image.open(p).convert('RGB')) for p in pos_batch]

            neg_aug1 = [self.transform_neg(Image.open(p).convert('RGB')) for p in neg_batch]
            neg_aug2 = [self.transform_pos(Image.open(p).convert('RGB')) for p in neg_batch]
            
            yield torch.stack(pos_aug1), torch.stack(pos_aug2), torch.stack(neg_aug1), torch.stack(neg_aug2)



